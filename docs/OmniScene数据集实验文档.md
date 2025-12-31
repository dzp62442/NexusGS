# OmniScene 数据集实验方案（NexusGS）

## 1. 背景与目标
OmniScene 是基于 nuScenes 的前馈式环视重建数据集。在 depthsplat 项目中，它通过 `DatasetOmniScene` 直接为 Encoder-Decoder 模型提供 `context/target` 视图，并依赖 Hydra 配置一次性完成全数据训练。本项目（NexusGS）属于逐场景优化式 3D Gaussian Splatting，需要将 OmniScene 的 bin 视作独立场景，逐个执行“数据预处理 → 场景初始化 → 迭代优化 → 渲染与指标评估”的流程。本文档的目标是：

1. 复用 depthsplat 的数据解析策略（尤其是 `load_conditions` 的路径替换与内参缩放逻辑），收敛为适合 NexusGS 的 per-scene 数据包。
2. 设计 `comp_svfgs` 子模块与 `output` 工作区，完成数据转换、加载与主流程调度。
3. 明确脚本入口、参数、运行顺序，使单个命令即可跑完整个 OmniScene val split（10 个 bin）的实验，为后续实现提供基准。
4. 在已有绝对深度（Metric3D-v2）的基础上，跳过光流生成/深度估计流程，直接复用官方深度来初始化点云。

## 2. depthsplat 的实现要点回顾
- **配置结构**：Hydra 配置 `config/dataset/omniscene.yaml` 绑定 `DatasetOmniScene`，再由 `+experiment=omniscene_*` 设定分辨率（112×200 或 224×400）、batchsize 等。训练入口统一为 `python -m src.main`。
- **数据划分**：`bins_train_3.2m.json` / `bins_val_3.2m.json` / `bins_dynamic_demo` 提供 token 列表；val 阶段通过 `self.bin_tokens = self.bin_tokens[:30000:3000][:10]` 选出十个 bin 进行验证。
- **样本内容**：每个 bin 的 `sensor_info` 自带 6 个环视摄像头在多个时间帧上的姿态。`context` 使用 key-frame 的 6 张图；`target` 再拼入 key-frame 之外的帧（每个摄像头取索引 `[1, 2]`），最终获得 18 张 supervision 图像。
- **加载细节**：`load_conditions` 负责 JPEG → 指定分辨率的 resize，并使用 `samples_param_small / sweeps_param_small` 的 JSON 重写内参。
- **主流程**：feed-forward 模型一次性读取 batch 中的多个 bin，直接在 GPU 上渲染预测。不存在 per-scene 重复初始化或磁盘缓存。

## 3. NexusGS 中的总体策略
为了让 NexusGS 支持 OmniScene，我们将整个流程拆分为“离线预处理 + 逐场景优化 + 渲染评估”三个阶段，核心设计如下：

1. **目录规划**：
   - `comp_svfgs/`：自定义实现，包括 `dataset_omniscene.py`、数据预处理脚本、运行调度器等。
   - `output/`：统一存放实验输出（预处理缓存、训练权重、渲染结果、指标）。项目本身已有 `output` 需求，提前创建目录。

2. **数据加载模块**（`comp_svfgs/dataset_omniscene.py`）：
   - 参考 depthsplat 与 SVF-GS 的 `DatasetOmniScene`/`nuScenesDataset`，实现 `OmniSceneDataset` 类，支持 `train/val/test/demo` 四种模式，默认 `mode='val'`，并沿用“前 30000 个 token 每 3000 取 1 个，再截取 10 个”的筛选方式。
   - `__getitem__` 需返回 `context` (6 张输入) 与 `target` (18 张评估) 的字典，内容包括图像 tensor、归一化内参、4×4 外参，以及 `bin_token` 元信息。
   - `load_conditions` 原样迁移 SVF-GS `data/transforms/loading.py` 中的实现，确保同步返回：
     - `rgb` 图像；
     - Metric3D-v2 绝对深度 `depths_m`（`samples_dptm_small/*.npy` 或 `sweeps_dptm_small/*.npy`）及其置信度 `confs_m`（`*_conf.npy`），用于生成有效像素掩码（统一规则：`depth_valid = conf > 0.3`）。
     - resize 后归一化的相机内参 `cks`（同 depthsplat 逻辑：`fx/fy/cx/cy` 随图像尺寸线性缩放后再除以宽/高）。

3. **数据预处理**：
   - 由于 NexusGS 优化流程依赖磁盘上的“场景目录”，我们将新增 `comp_svfgs/preprocess_omniscene.py`。该脚本调用 `OmniSceneDataset(mode='val')`，遍历 10 个 bin，将 `context/target` 数据保存到 `output/omniscene_preprocessed/<rank>_<token>/`。
   - 每个场景目录包含：
     - `context/images/000.png`~`005.png` 与 `context/cameras.json`（记录对应外参、归一化内参、图像原尺寸、近远平面等）。
     - `context/depth_metric.npy`（shape=[6,H,W]）与 `context/depth_valid.npy`（根据 `conf > 0.3` 生成的布尔掩码），用于直接写入 `CameraInfo.flow_depth`。
     - `target/images/*.png`（共 18 张）、`target/depth_metric.npy`、`target/depth_valid.npy` 与 `target/cameras.json`。
     - `meta.json`：保存 `bin_token`、原始数据根目录、分辨率等信息，便于追踪。
   - 预处理时同步生成 `context/cams.npz` 与 `target/cams.npz`，字段包括：
     - `extrinsics` (N,4,4)、`intrinsics` (N,3,3)、`image_paths`、`image_size`、`near`、`far`；
     - `depth_metric`（单位：米）与 `depth_valid`（0/1，阈值 0.3）；
     - 数据是否归一化的标志位 `intrinsics_normalized`（默认 true）。
   - 首次运行写入缓存，后续执行若检测到 `cams.npz` 和对应 `.png/.npy` 均存在且 `meta.json` 中的 `hash`/`timestamp` 未变化，则跳过。

4. **场景装载与训练接口**：
   - 在 `comp_svfgs` 中再提供 `omniscene_scene.py`（或扩展 `dataset_omniscene.py`），读取缓存目录并转换成 NexusGS 所需的 `CameraInfo` 列表。关键是把 `context` 中的 6 个视图转换为 `Scene` 初始化所需的 `train_cameras`，`target` 中的 18 个视图转化为 `test_cameras`/`eval_cameras`。
   - 引入新的 `dataset_type='omniscene'`：在 `scene/dataset_readers.py` 中注册新的回调，遇到该类型时直接读取 `output/omniscene_preprocessed/<scene>/context|target` 生成 `SceneInfo`，避免依赖 COLMAP / Blender 目录结构。
   - **跳过光流/深度计算**：对于 `dataset_type='omniscene'`：
     - `cameraList_from_camInfos` 应从缓存里读取 `depth_metric` 与 `depth_valid`，并调用 helper（例如 `camera.set_flow_depth_from_metric(depth, depth_valid)`）直接填充 `cam.flow_depth` 与 `cam.flow_depth_mask`。
     - `Scene.__init__` 中默认的 `compute_depth_by_flow(...)` 调用在检测到 `args.dataset_type == 'omniscene'` 时可以直接跳过（或变成空函数），因为深度已随缓存提供。
     - `construct_pcd` 原封不动使用 `cam.flow_depth`/`flow_depth_mask`，从 Metric 深度生成点云，从而完全摆脱光流依赖。
   - 训练参数设置：
     - `--images` 固定为生成目录下的 `images`，但由于我们手动控制了尺寸，`-r` 保持为 1，且无需多尺度金字塔。
     - `--n_views=6`，每个输入视图都作为训练相机。
     - 迭代次数沿用默认配置（如 30k），后续可在脚本中提供 CLI 覆盖。

5. **运行脚本**（`scripts/run_omniscene.py`）：
   - 入口为 Python 脚本，主要流程：
     1. 解析参数（模式、分辨率、是否跳过渲染/评估、GPU ID 等）。
     2. 调用预处理逻辑：若 `output/omniscene_preprocessed/<scene>` 不存在，执行转换；存在则直接跳过。
     3. 对每个场景构造 `source_path`（指向该场景目录）、`model_path=output/omniscene_results/<scene>`，依次调用 `train.py`、`render.py`、`metrics.py`。训练完成后自动渲染 18 个 target 视角并计算指标。
     4. 记录日志（可写入 `output/omniscene_results/summary.jsonl`），包括时间、 PSNR/SSIM、输出路径。
   - 由于每个 bin 视为单场景，脚本将顺序运行 10 次；后续可加 `--scene_indices` 仅处理部分。

6. **结果组织**：
   - 训练阶段生成的 ply/ckpt 存在 `output/omniscene_results/<scene>/`。
   - 渲染输出与指标（`rendered/`、`metrics.json`）同样放在该目录内，方便比对与归档。
   - 对于与自研方法（depthsplat/com_svfgs）的对比，可在文档中记录统一的输出命名规则，确保后续调参具有可追踪性。

## 4. 数据处理流程细化
1. **原始数据定位**：假定 OmniScene 数据放在 `datasets/omniscene/`。脚本将通过环境变量或默认路径查找：
   - `bins_*.json`、`bin_infos_3.2m/*.pkl`、`scene_mapping.pkl`；
   - RGB：`samples_small/*.jpg`、`sweeps_small/*.jpg`；
   - Metric depth：`samples_dptm_small/*_dpt.npy`、`sweeps_dptm_small/*_dpt.npy`；
   - Metric depth 置信度：`*_conf.npy`，用于生成深度有效性掩码；
   - 相机参数：`samples_param_small/*.json`、`sweeps_param_small/*.json`。
2. **加载与 resize**：
   - `load_conditions` 接收 `resolution=(112, 200)`（默认）或 `(224, 400)`；若传参 `--resolution 224x400` 则切换。
   - resize 算法与 depthsplat/SVF-GS 一致：`fx/fy/cx/cy` 先按宽高缩放，再除以输出宽高进行归一化，保证与 NexusGS 的 `CameraInfo` 期望一致。
   - 深度与置信度在 resize 时使用双线性插值，随后按 `conf > 0.3` 生成 `depth_valid`。
3. **外参处理**：
   - 使用 depthsplat 版本的 `load_info`（不执行 `flip_yz`），保持与当前 NexusGS 坐标系一致。
   - 归一化后的 `c2w`、`w2c` 与 camera radius 一同存储在 `context/target/cameras.json`。
4. **缓存格式**（详细）：
   - `context/cameras.json`：
     ```json
     {
       "views": [
         {
           "image": "images/000.png",
           "depth_metric": "depth_metric/000.npy",
           "extrinsic": [[...],[...],[...],[...]],
           "intrinsic": [[...],[...],[...]],
           "width": 200,
           "height": 112
         },
         ...
       ],
       "near": 0.1,
       "far": 1000.0
     }
     ```
   - `context/cams.npz` 与 `target/cams.npz`：dict 形式，字段 `extrinsics`、`intrinsics`、`images`, `depth_metric`, `depth_valid`, `near`, `far`, `resolution`。
   - 图像与深度文件分开放置，命名为 `depth_metric/000.npy` 等，以便脚本判断是否存在。
   - 预处理完成后写入 `meta.json`，记录 `bin_token`、`mode`、`resolution`、`version_hash`（由输入路径与文件大小散列而来），供脚本检测缓存是否过期。

## 5. 逐场景优化流程
1. **初始化**：`Scene(args, gaussians, shuffle=False)` 在 `dataset_type='omniscene'` 时直接读取缓存，构造 `train_cameras`（6 个输入）和 `test/eval_cameras`（18 个输出）。初始点云通过 `construct_pcd` 需要 GT depth，因此：
   - 使用 Metric3D 绝对深度直接填入 `cam.flow_depth`（单位为米），并根据置信度生成 `flow_depth_mask`（固定阈值 `conf > 0.3`）。
   - `construct_pcd` 读取这些深度/掩码即可恢复稠密点云；不再依赖光流文件。
2. **训练**：`train.py` 运行 30k iter，`--n_views=6`，`-r 1`，`--images images`（默认 `images_8` 无需使用，但 `Scene` 不再依赖该字段）。脚本根据 `--model_path output/omniscene_results/<scene>` 自动记录日志，并在 30k iteration 时保存 ply。
3. **渲染与评估**：
   - 渲染阶段将 `--iteration` 固定为 30000，`--render_depth` 以便观察几何。
   - `metrics.py` 使用 target 视角的真值图像，输出 per-scene 指标（PSNR/SSIM/LPIPS）。

## 6. 运行命令示例
```bash
python scripts/run_omniscene.py \
    --mode val \
    --dataset-root ~/datasets/omniscene \
    --resolution 112x200 \
    --bin-limit 10 \
    --gpus 0 \
    --iterations 30000
```
- 首次运行会在 `output/omniscene_preprocessed/` 生成 `01_<token>/...` 等目录，并按顺序处理 10 个场景。
- 再次运行时若检测到 `scene_dir/meta.json` 且哈希/时间戳未变化，则跳过预处理直接进入训练。
- 渲染与评估结果位于 `output/omniscene_results/<scene>/render/` 与 `metrics.json`。

## 7. feed-forward 与逐场景优化的差异总结
| 维度 | depthsplat（前馈） | NexusGS（逐场景优化） | 适配策略 |
| --- | --- | --- | --- |
| 数据装载 | Hydra Dataset → batch tensor | 每次只处理一个场景目录 | 预处理生成场景目录，`Scene` 逐个加载 |
| 输入视图 | `context` 一次性参与网络前传 | 训练 loop 中随机/循环采样 | 将 6 张 context 直接注册为 `train_cameras` |
| 输出 supervision | 同批 `target` 用于 loss | 训练后渲染比较 | 将 18 张 target 保存为 eval/test 相机，优化后使用 `render.py` + `metrics.py` 计算指标 |
| 深度使用 | 仅在网络内部估/监督 | 需要初始化点云 | 预处理阶段加载 Metric3D depth → 写入 `cam.flow_depth`，跳过光流流程 |
| 运行脚本 | `python -m src.main +experiment=...` | `scripts/run_omniscene.py` 循环调用 `train.py`/`render.py`/`metrics.py` | 统一入口、支持断点续跑 |

## 8. 下一步
1. 完成 `comp_svfgs/dataset_omniscene.py` 与数据缓存脚本，实现四种模式加载与 `load_conditions` 复制。
2. 拓展 `scene/dataset_readers.py` 及相关工具，使 `dataset_type='omniscene'` 可生成 `CameraInfo` 列表。
3. 开发 `scripts/run_omniscene.py`，串联预处理、训练、渲染、评估。
4. 在少量场景（例如 `01_<token>`）上进行冒烟测试，验证指标输出；若通过，再扩大到完整 val split。
