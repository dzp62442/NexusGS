# NexusGS OmniScene center150 实验

## 目标与默认协议

NexusGS 将每个 OmniScene bin 视为一个独立场景：6 张中央帧环视图像用于优化，12 张相邻帧图像与 6 张中央帧图像组成 18 张 target 图像用于评估。

本分支默认运行 `center150`：从 nuScenes val 的 150 个场景中各取一个中央 bin，并在 1k、5k、10k 次迭代时分别评估。默认分辨率为 112×200，总迭代数为 10k。

center150 清单由 SVF-GS 生成。本项目不包含生成逻辑，只读取：

```text
datasets/omniscene/interp_12Hz_trainval/bins_center150_v1.json
```

加载前会同时读取 `bins_val_3.2m.json`，逐场景复算 lower median，并校验：

- val 恰好包含 150 个场景分组；
- `adjacent_bins` 展平后与 `bins` 完全一致；
- 每个场景内 bin 从 0 开始连续且有序；
- center150 的 150 个 token 与复算结果按顺序逐项相同且无重复；
- 对应的 150 个 `bin_infos_3.2m/*.pkl` 全部存在。

任一条件不满足都会直接报错，不会在 NexusGS 内生成或修补清单。

## 启动

激活项目环境后，默认实验只需要：

```bash
python scripts/run_omniscene.py
```

常用覆写示例：

```bash
# 只运行第 1、7、20 个样本
python scripts/run_omniscene.py --scene-indices 1,7,20

# 使用另一张 GPU 和独立结果目录
python scripts/run_omniscene.py --gpus 1 --results-root output/center150_ablation

# 覆写 train.py 的非调度参数；未知参数会透传给 train.py
python scripts/run_omniscene.py --depth_weight 0.1 --lambda_dssim 0.15
```

迭代总数、评估点、保存点和 checkpoint 点由调度器统一管理。若要更改协议，应同时覆写总迭代数和最后一个评估点，并使用新的结果目录，例如：

```bash
python scripts/run_omniscene.py \
  --iterations 15000 \
  --eval-iterations 1000 5000 10000 15000 \
  --results-root output/omniscene_results/center150_15k
```

默认结果目录已经按 center150 隔离：

```text
output/omniscene_preprocessed/center150/
output/omniscene_results/center150/
```

旧的 10-bin val 模式仍可显式调用：

```bash
python scripts/run_omniscene.py --mode val
```

## 里程碑评估与耗时

每个里程碑都会生成：

```text
<scene>/point_cloud/iteration_<iter>/point_cloud.ply
<scene>/chkpnt<iter>.pth
<scene>/test/ours_<iter>/renders/*.png
<scene>/test/ours_<iter>/gt/*.png
<scene>/evaluation/iteration_<iter>.json
```

结构化评估文件记录 18 个 target 视图上的 PSNR、SSIM、LPIPS、L1，以及到该里程碑为止的累计训练耗时。计时在 CUDA 同步后取值，并排除完整评估、PLY 保存和 checkpoint 写盘开销，因此不同里程碑反映的是优化本身的累计耗时，而不是评估图片与模型落盘耗时。

评估会保存并恢复 Torch CPU/CUDA 随机状态；对于 OmniScene，增加评估点不会额外跳过致密化步骤。

## 断点续跑

调度器按以下顺序恢复：

1. 若 `scene_complete.json` 与全部里程碑产物均通过校验，快速跳过该场景；
2. 若场景未完成但存在里程碑 checkpoint，从最新 checkpoint 继续；
3. 若没有 checkpoint，从头训练该场景。

checkpoint 包含高斯参数、优化器状态、致密化辅助状态、Python/Torch/CUDA 随机状态、剩余训练视角队列和累计训练耗时。场景完成标记与指标文件均采用临时文件加原子替换，避免中断时把半成品误判为完成。

同一结果根目录会固定一份 `experiment_config.json`。数据清单、分辨率、迭代协议或透传训练参数发生变化时，必须指定新的 `--results-root`，以免混合不可比实验。

## 自动汇总

每轮结束都会写入：

```text
output/omniscene_results/center150/center150_progress.json
```

当且仅当全部 150 个场景通过完整性校验后，自动生成：

```text
center150_metrics_summary.json
center150_metrics_summary.txt
```

汇总对 150 个样本等权平均，分别报告 1k、5k、10k 时的 PSNR、SSIM、LPIPS、L1 和累计训练耗时。JSON 同时保留每个样本的原始记录，便于复核与后续统计。

## 数据与相机约定

- RGB 从 `samples_small` / `sweeps_small` 加载；Metric3D-v2 绝对深度和置信度从 `*_dpt.npy` / `*_conf.npy` 加载。
- 深度有效掩码默认为 `confidence > 0.3`。
- `sensor2lidar_transform` 直接作为 OpenCV 相机到关键帧 LiDAR 的 `c2w`，并以 `w2c = inverse(c2w)` 构造 NexusGS 的 `R/T`。
- 预处理格式版本用于拒绝旧的错误坐标缓存。
- 本项目既有的全局统一内参假设保持不变。
