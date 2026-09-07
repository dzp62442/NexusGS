import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# 指标回填必须在任何 torch 模块导入前屏蔽 GPU。
if "--metrics-only" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

from comp_svfgs.dataset_omniscene import (  # noqa: E402
    CENTER150_SAMPLE_COUNT,
    OmniSceneDataset,
)
from comp_svfgs.preprocess_omniscene import (  # noqa: E402
    OMNISCENE_PREPROCESSED_FORMAT_VERSION,
    _parse_resolution,
    preprocess_scene,
)


EXPERIMENT_FORMAT_VERSION = 1
EVALUATION_FORMAT_VERSION = 2
LEGACY_EVALUATION_FORMAT_VERSION = 1
SUMMARY_FORMAT_VERSION = 2
METRIC_KEYS = ("psnr", "ssim", "lpips", "l1")
ALL_18_VIEW_GROUP = "all_18_views"
NOVEL_12_VIEW_GROUP = "novel_12_views"
NOVEL_VIEW_COUNT = 12
MANAGED_TRAIN_ARGUMENTS = {
    "-s",
    "--source_path",
    "-m",
    "--model_path",
    "--dataset_type",
    "--n_views",
    "--images",
    "-r",
    "--resolution",
    "--iterations",
    "--densify_until_iter",
    "--position_lr_max_steps",
    "--test_iterations",
    "--save_iterations",
    "--checkpoint_iterations",
    "--start_checkpoint",
    "--full_eval_metrics",
    "--eval",
}


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(text, encoding="utf-8")
    os.replace(temporary_path, path)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _load_meta(scene_dir: Path) -> dict:
    meta_path = scene_dir / "meta.json"
    if not meta_path.is_file():
        return {}
    try:
        return load_json(meta_path)
    except (json.JSONDecodeError, ValueError):
        return {}


def preprocessed_scene_complete(
    scene_dir: Path,
    bin_token: str,
    resolution: tuple[int, int],
    valid_threshold: float,
    mode: str,
) -> bool:
    meta = _load_meta(scene_dir)
    res = meta.get("resolution", {})
    if not (
        meta.get("format_version") == OMNISCENE_PREPROCESSED_FORMAT_VERSION
        and meta.get("bin_token") == bin_token
        and meta.get("mode") == mode
        and res.get("height") == resolution[0]
        and res.get("width") == resolution[1]
        and meta.get("valid_threshold") == valid_threshold
        and meta.get("num_context") == 6
        and meta.get("num_target") == 18
    ):
        return False
    required_files = []
    for block_name, expected_views in (("context", 6), ("target", 18)):
        block_dir = scene_dir / block_name
        required_files.extend(
            block_dir / filename
            for filename in (
                "cams.npz",
                "cameras.json",
                "depth_metric.npy",
                "depth_valid.npy",
            )
        )
        try:
            views = load_json(block_dir / "cameras.json").get("views")
        except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
            return False
        if not isinstance(views, list) or len(views) != expected_views:
            return False
        for view in views:
            if not isinstance(view, dict) or not isinstance(view.get("image"), str):
                return False
            required_files.append(block_dir / view["image"])
    return all(nonempty_file(path) for path in required_files)


def ensure_scene_preprocessed(
    sample: dict,
    scene_dir: Path,
    resolution: tuple[int, int],
    valid_threshold: float,
    mode: str,
) -> None:
    if preprocessed_scene_complete(
        scene_dir, sample["bin_token"], resolution, valid_threshold, mode
    ):
        return
    print(f"[Preprocess] {sample['bin_token']}")
    preprocess_scene(sample, scene_dir, resolution, mode, valid_threshold)
    if not preprocessed_scene_complete(
        scene_dir, sample["bin_token"], resolution, valid_threshold, mode
    ):
        raise RuntimeError(f"预处理完成校验失败: {scene_dir}")


def ordered_target_image_names(scene_dir: Path) -> list[str]:
    cameras_path = scene_dir / "target" / "cameras.json"
    if not cameras_path.is_file():
        return []
    try:
        cameras = load_json(cameras_path).get("views")
    except (json.JSONDecodeError, OSError, ValueError):
        return []
    if not isinstance(cameras, list):
        return []
    names = [
        f"{view['image_name']}.png"
        for view in cameras
        if isinstance(view, dict) and isinstance(view.get("image_name"), str)
    ]
    return names if len(names) == len(cameras) and len(set(names)) == len(names) else []


def expected_target_image_names(scene_dir: Path) -> set[str]:
    return set(ordered_target_image_names(scene_dir))


def validate_metrics(metrics: object, description: str) -> dict:
    if not isinstance(metrics, dict):
        raise ValueError(f"{description} 缺少指标字典")
    for key in METRIC_KEYS:
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{description} 的指标 {key} 非法")
    if metrics["lpips"] < 0.0 or metrics["l1"] < 0.0:
        raise ValueError(f"{description} 的指标范围非法")
    return metrics


def evaluation_has_view_groups(record: dict) -> bool:
    view_groups = record.get("view_groups")
    if not isinstance(view_groups, dict):
        return False
    all_group = view_groups.get(ALL_18_VIEW_GROUP)
    novel_group = view_groups.get(NOVEL_12_VIEW_GROUP)
    if not isinstance(all_group, dict) or not isinstance(novel_group, dict):
        return False
    try:
        validate_metrics(all_group.get("metrics"), ALL_18_VIEW_GROUP)
        validate_metrics(novel_group.get("metrics"), NOVEL_12_VIEW_GROUP)
    except ValueError:
        return False
    return (
        all_group.get("num_views") == record.get("num_views")
        and novel_group.get("num_views") == NOVEL_VIEW_COUNT
        and all_group["metrics"] == record.get("metrics")
    )


def load_evaluation(model_dir: Path, iteration: int, expected_num_views: int) -> dict:
    evaluation_path = model_dir / "evaluation" / f"iteration_{iteration}.json"
    record = load_json(evaluation_path)
    if (
        record.get("format_version")
        not in (LEGACY_EVALUATION_FORMAT_VERSION, EVALUATION_FORMAT_VERSION)
        or record.get("iteration") != iteration
        or record.get("split") != "test"
        or record.get("num_views") != expected_num_views
    ):
        raise ValueError(f"评估记录元数据不匹配: {evaluation_path}")
    validate_metrics(record.get("metrics"), str(evaluation_path))
    if record.get("format_version") == EVALUATION_FORMAT_VERSION:
        if not evaluation_has_view_groups(record):
            raise ValueError(f"评估记录缺少完整视角分组: {evaluation_path}")
    training_time = record.get("training_time_seconds")
    if not isinstance(training_time, (int, float)) or not math.isfinite(training_time):
        raise ValueError(f"训练耗时非法: {evaluation_path}")
    if training_time < 0.0:
        raise ValueError(f"训练耗时为负数: {evaluation_path}")
    return record


def compute_png_metrics_cpu(
    render_dir: Path,
    gt_dir: Path,
    image_names: list[str],
    lpips_metric,
    batch_size: int,
) -> dict[str, float]:
    import torch
    from PIL import Image
    from torchvision.transforms.functional import to_tensor

    from utils.loss_utils import ssim

    if not image_names:
        raise ValueError("CPU 指标计算没有输入图像")
    renders = []
    ground_truths = []
    for image_name in image_names:
        with Image.open(render_dir / image_name) as image:
            renders.append(to_tensor(image.convert("RGB")))
        with Image.open(gt_dir / image_name) as image:
            ground_truths.append(to_tensor(image.convert("RGB")))
    renders = torch.stack(renders)
    ground_truths = torch.stack(ground_truths)

    with torch.inference_mode():
        l1_value = torch.abs(renders - ground_truths).flatten(1).mean(1).mean()
        mse_values = ((renders - ground_truths) ** 2).flatten(1).mean(1)
        psnr_value = (20 * torch.log10(1.0 / torch.sqrt(mse_values))).mean()
        ssim_value = ssim(renders, ground_truths)

        lpips_values = []
        for start in range(0, len(image_names), batch_size):
            render_batch = renders[start : start + batch_size]
            gt_batch = ground_truths[start : start + batch_size]
            render_features = lpips_metric.net(render_batch)
            gt_features = lpips_metric.net(gt_batch)
            layer_values = [
                layer((render_feature - gt_feature) ** 2).mean((2, 3)).flatten()
                for layer, render_feature, gt_feature in zip(
                    lpips_metric.lin, render_features, gt_features
                )
            ]
            lpips_values.append(torch.stack(layer_values).sum(0))
        lpips_value = torch.cat(lpips_values).mean()

    metrics = {
        "psnr": psnr_value.item(),
        "ssim": ssim_value.item(),
        "lpips": lpips_value.item(),
        "l1": l1_value.item(),
    }
    return validate_metrics(metrics, "CPU PNG 指标")


def backfill_missing_view_groups(
    dataset: OmniSceneDataset,
    output_root: Path,
    results_root: Path,
    eval_iterations: list[int],
    experiment_fingerprint: str,
    cpu_threads: int,
    batch_size: int,
) -> int:
    tasks = []
    touched_scenes = set()
    for index, bin_token in enumerate(dataset.bin_tokens):
        name = scene_name(index, len(dataset), bin_token)
        scene_dir = output_root / name
        model_dir = results_root / name
        if not scene_complete(
            scene_dir, model_dir, eval_iterations, experiment_fingerprint, bin_token
        ):
            continue
        ordered_names = ordered_target_image_names(scene_dir)
        if len(ordered_names) != 18:
            raise RuntimeError(f"{name} 的 target 视图数不是 18")
        for iteration in eval_iterations:
            record = load_evaluation(model_dir, iteration, len(ordered_names))
            if not evaluation_has_view_groups(record):
                tasks.append(
                    (index, bin_token, name, scene_dir, model_dir, iteration, ordered_names)
                )

    if not tasks:
        print("[Metrics] 18 路与 12 路指标均已存在，无需回填")
        return 0

    import torch
    from lpipsPyTorch import LPIPS

    if cpu_threads <= 0 or batch_size <= 0:
        raise ValueError("CPU 线程数和指标 batch size 必须为正数")
    if "--metrics-only" in sys.argv and torch.cuda.is_available():
        raise RuntimeError("--metrics-only 已要求屏蔽 GPU，但当前进程仍可访问 CUDA")
    torch.set_num_threads(cpu_threads)
    lpips_metric = LPIPS(net_type="vgg").cpu().eval()
    print(
        f"[Metrics] 使用 CPU 回填 {len(tasks)} 个里程碑的前 12 路新视角指标 "
        f"(threads={cpu_threads}, batch={batch_size})"
    )

    for task_index, (
        index,
        bin_token,
        name,
        scene_dir,
        model_dir,
        iteration,
        ordered_names,
    ) in enumerate(tasks, 1):
        evaluation_path = model_dir / "evaluation" / f"iteration_{iteration}.json"
        record = load_evaluation(model_dir, iteration, len(ordered_names))
        original_training_time = record["training_time_seconds"]
        method_dir = model_dir / "test" / f"ours_{iteration}"
        novel_metrics = compute_png_metrics_cpu(
            method_dir / "renders",
            method_dir / "gt",
            ordered_names[:NOVEL_VIEW_COUNT],
            lpips_metric,
            batch_size,
        )
        all_metrics = dict(record["metrics"])
        record["format_version"] = EVALUATION_FORMAT_VERSION
        record["view_groups"] = {
            ALL_18_VIEW_GROUP: {
                "num_views": len(ordered_names),
                "metrics": all_metrics,
                "source": "training_in_memory",
            },
            NOVEL_12_VIEW_GROUP: {
                "num_views": NOVEL_VIEW_COUNT,
                "metrics": novel_metrics,
                "source": "saved_png_cpu_postprocess",
            },
        }
        if record["training_time_seconds"] != original_training_time:
            raise RuntimeError(f"禁止修改已有训练耗时: {evaluation_path}")
        atomic_write_json(evaluation_path, record)
        touched_scenes.add((index, bin_token, name, scene_dir, model_dir))
        if task_index % 25 == 0 or task_index == len(tasks):
            print(f"[Metrics] {task_index}/{len(tasks)}")

    for index, bin_token, name, scene_dir, model_dir in touched_scenes:
        write_scene_completion(
            scene_dir,
            model_dir,
            index + 1,
            bin_token,
            eval_iterations,
            experiment_fingerprint,
        )
    return len(tasks)


def milestone_complete(
    scene_dir: Path,
    model_dir: Path,
    iteration: int,
    expected_names: set[str],
) -> bool:
    if not expected_names:
        return False
    try:
        load_evaluation(model_dir, iteration, len(expected_names))
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        return False

    required_files = (
        model_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply",
        model_dir / f"chkpnt{iteration}.pth",
    )
    if not all(nonempty_file(path) for path in required_files):
        return False

    method_dir = model_dir / "test" / f"ours_{iteration}"
    for subdir_name in ("renders", "gt"):
        subdir = method_dir / subdir_name
        if not subdir.is_dir():
            return False
        actual_names = {path.name for path in subdir.iterdir() if path.is_file()}
        if actual_names != expected_names:
            return False
        if not all(nonempty_file(subdir / name) for name in expected_names):
            return False
    return True


def scene_complete(
    scene_dir: Path,
    model_dir: Path,
    eval_iterations: list[int],
    experiment_fingerprint: str,
    bin_token: str,
) -> bool:
    completion_path = model_dir / "scene_complete.json"
    try:
        completion = load_json(completion_path)
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        return False
    if (
        completion.get("format_version") != EXPERIMENT_FORMAT_VERSION
        or completion.get("experiment_fingerprint") != experiment_fingerprint
        or completion.get("bin_token") != bin_token
        or completion.get("eval_iterations") != eval_iterations
    ):
        return False

    expected_names = expected_target_image_names(scene_dir)
    records = []
    for iteration in eval_iterations:
        if not milestone_complete(scene_dir, model_dir, iteration, expected_names):
            return False
        records.append(load_evaluation(model_dir, iteration, len(expected_names)))
    training_times = [record["training_time_seconds"] for record in records]
    return training_times == sorted(training_times)


def write_scene_completion(
    scene_dir: Path,
    model_dir: Path,
    scene_index: int,
    bin_token: str,
    eval_iterations: list[int],
    experiment_fingerprint: str,
) -> None:
    expected_names = expected_target_image_names(scene_dir)
    records = {
        str(iteration): load_evaluation(model_dir, iteration, len(expected_names))
        for iteration in eval_iterations
    }
    training_times = [
        records[str(iteration)]["training_time_seconds"] for iteration in eval_iterations
    ]
    if training_times != sorted(training_times):
        raise RuntimeError(f"累计训练耗时没有单调递增: {model_dir}")
    atomic_write_json(
        model_dir / "scene_complete.json",
        {
            "format_version": EXPERIMENT_FORMAT_VERSION,
            "experiment_fingerprint": experiment_fingerprint,
            "scene_index": scene_index,
            "bin_token": bin_token,
            "eval_iterations": eval_iterations,
            "evaluations": records,
        },
    )


def find_latest_checkpoint(model_dir: Path, total_iterations: int) -> tuple[int, Path] | None:
    candidates = []
    for checkpoint_path in model_dir.glob("chkpnt*.pth"):
        suffix = checkpoint_path.stem.removeprefix("chkpnt")
        if suffix.isdigit() and nonempty_file(checkpoint_path):
            iteration = int(suffix)
            if iteration <= total_iterations:
                candidates.append((iteration, checkpoint_path))
    return max(candidates, default=None, key=lambda item: item[0])


def run_command(command: list[str], env: dict[str, str]) -> None:
    print("[Command]", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT_DIR, env=env)


def parse_scene_indices(value: str | None, dataset_size: int, bin_limit: int | None) -> list[int]:
    if value:
        indices = []
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            index = int(item) - 1
            if index < 0 or index >= dataset_size:
                raise ValueError(f"scene index 超出范围: {item}，有效范围为 1..{dataset_size}")
            if index not in indices:
                indices.append(index)
        if not indices:
            raise ValueError("--scene-indices 没有给出有效索引")
        return indices
    count = dataset_size if bin_limit is None else min(bin_limit, dataset_size)
    if count <= 0:
        raise ValueError("--bin-limit 必须为正数")
    return list(range(count))


def validate_train_overrides(train_overrides: list[str]) -> None:
    for token in train_overrides:
        option = token.split("=", 1)[0]
        if option in MANAGED_TRAIN_ARGUMENTS:
            raise ValueError(
                f"训练参数 {option} 由调度器统一管理，请使用对应的 run_omniscene.py 参数覆写"
            )


def experiment_configuration(args, dataset: OmniSceneDataset, resolution, train_overrides) -> dict:
    manifest_json = json.dumps(dataset.bin_tokens, ensure_ascii=False, separators=(",", ":"))
    return {
        "format_version": EXPERIMENT_FORMAT_VERSION,
        "mode": args.mode,
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "data_version": dataset.data_version,
        "manifest_sha256": hashlib.sha256(manifest_json.encode("utf-8")).hexdigest(),
        "num_samples": len(dataset),
        "resolution": {"height": resolution[0], "width": resolution[1]},
        "valid_threshold": args.valid_threshold,
        "iterations": args.iterations,
        "eval_iterations": args.eval_iterations,
        "densify_until_iter": args.densify_until_iter,
        "position_lr_max_steps": args.position_lr_max_steps,
        "train_overrides": train_overrides,
    }


def ensure_experiment_configuration(results_root: Path, configuration: dict) -> str:
    serialized = json.dumps(configuration, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    config_path = results_root / "experiment_config.json"
    if config_path.exists():
        existing = load_json(config_path)
        if existing != configuration:
            raise RuntimeError(
                f"结果目录已有不同实验配置: {config_path}\n"
                "请改用新的 --results-root，避免混合不可比结果。"
            )
    else:
        atomic_write_json(config_path, configuration)
    return fingerprint


def scene_name(index: int, dataset_size: int, bin_token: str) -> str:
    width = max(2, len(str(dataset_size)))
    return f"{index + 1:0{width}d}_{bin_token}"


def build_train_command(
    args,
    scene_dir: Path,
    model_dir: Path,
    checkpoint: tuple[int, Path] | None,
    train_overrides: list[str],
) -> list[str]:
    command = [
        sys.executable,
        "train.py",
        "--source_path",
        str(scene_dir),
        "--model_path",
        str(model_dir),
        "--dataset_type",
        "omniscene",
        "--n_views",
        "6",
        "--images",
        "images",
        "--resolution",
        "1",
        "--iterations",
        str(args.iterations),
        "--densify_until_iter",
        str(args.densify_until_iter),
        "--position_lr_max_steps",
        str(args.position_lr_max_steps),
        "--test_iterations",
        *map(str, args.eval_iterations),
        "--save_iterations",
        *map(str, args.eval_iterations),
        "--checkpoint_iterations",
        *map(str, args.eval_iterations),
        "--full_eval_metrics",
        "--eval",
    ]
    if checkpoint is not None:
        command.extend(["--start_checkpoint", str(checkpoint[1])])
    command.extend(train_overrides)
    return command


def process_scene(
    args,
    dataset: OmniSceneDataset,
    index: int,
    resolution: tuple[int, int],
    output_root: Path,
    results_root: Path,
    experiment_fingerprint: str,
    train_overrides: list[str],
    env: dict[str, str],
) -> None:
    bin_token = dataset.bin_tokens[index]
    name = scene_name(index, len(dataset), bin_token)
    scene_dir = output_root / name
    model_dir = results_root / name

    if scene_complete(
        scene_dir,
        model_dir,
        args.eval_iterations,
        experiment_fingerprint,
        bin_token,
    ):
        print(f"[Skip] {index + 1}/{len(dataset)} {bin_token}")
        return

    if not preprocessed_scene_complete(
        scene_dir, bin_token, resolution, args.valid_threshold, args.mode
    ):
        sample = dataset[index]
        ensure_scene_preprocessed(
            sample, scene_dir, resolution, args.valid_threshold, args.mode
        )
    model_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = find_latest_checkpoint(model_dir, args.iterations)
    expected_names = expected_target_image_names(scene_dir)
    if checkpoint is not None:
        incomplete_saved_milestones = [
            iteration
            for iteration in args.eval_iterations
            if iteration <= checkpoint[0]
            and not milestone_complete(scene_dir, model_dir, iteration, expected_names)
        ]
        if incomplete_saved_milestones:
            raise RuntimeError(
                f"{name} 的 checkpoint {checkpoint[0]} 之前存在不完整里程碑: "
                f"{incomplete_saved_milestones}。请将该场景结果移到备份目录后重跑。"
            )
    if checkpoint is None or checkpoint[0] < args.iterations:
        if checkpoint is not None:
            print(f"[Resume] {name} from iteration {checkpoint[0]}")
        train_command = build_train_command(
            args, scene_dir, model_dir, checkpoint, train_overrides
        )
        run_command(train_command, env)

    for iteration in args.eval_iterations:
        if not milestone_complete(scene_dir, model_dir, iteration, expected_names):
            raise RuntimeError(f"{name} 的 iteration {iteration} 产物校验失败")
    write_scene_completion(
        scene_dir,
        model_dir,
        index + 1,
        bin_token,
        args.eval_iterations,
        experiment_fingerprint,
    )
    print(f"[Done] {index + 1}/{len(dataset)} {bin_token}")


def aggregate_center150(
    dataset: OmniSceneDataset,
    output_root: Path,
    results_root: Path,
    eval_iterations: list[int],
    experiment_fingerprint: str,
) -> bool:
    completed_records = []
    pending = []
    for index, bin_token in enumerate(dataset.bin_tokens):
        name = scene_name(index, len(dataset), bin_token)
        scene_dir = output_root / name
        model_dir = results_root / name
        if not scene_complete(
            scene_dir, model_dir, eval_iterations, experiment_fingerprint, bin_token
        ):
            pending.append({"scene_index": index + 1, "bin_token": bin_token})
            continue
        expected_names = expected_target_image_names(scene_dir)
        evaluations = {
            str(iteration): load_evaluation(model_dir, iteration, len(expected_names))
            for iteration in eval_iterations
        }
        if not all(evaluation_has_view_groups(record) for record in evaluations.values()):
            pending.append(
                {
                    "scene_index": index + 1,
                    "bin_token": bin_token,
                    "reason": "missing_view_group_metrics",
                }
            )
            continue
        completed_records.append(
            {
                "scene_index": index + 1,
                "bin_token": bin_token,
                "scene_name": name,
                "evaluations": evaluations,
            }
        )

    atomic_write_json(
        results_root / "center150_progress.json",
        {
            "format_version": EXPERIMENT_FORMAT_VERSION,
            "completed_samples": len(completed_records),
            "total_samples": len(dataset),
            "pending_samples": pending,
        },
    )
    if pending:
        print(f"[Progress] center150 {len(completed_records)}/{len(dataset)}，尚未生成最终汇总")
        return False

    averages = {}
    for iteration in eval_iterations:
        iteration_records = [
            record["evaluations"][str(iteration)] for record in completed_records
        ]
        view_group_averages = {
            group_name: {
                key: sum(
                    record["view_groups"][group_name]["metrics"][key]
                    for record in iteration_records
                )
                / len(iteration_records)
                for key in METRIC_KEYS
            }
            for group_name in (ALL_18_VIEW_GROUP, NOVEL_12_VIEW_GROUP)
        }
        # 顶层指标继续表示原有 18 路均值，保持旧分析脚本兼容。
        averages[str(iteration)] = dict(view_group_averages[ALL_18_VIEW_GROUP])
        averages[str(iteration)]["view_groups"] = view_group_averages
        averages[str(iteration)]["training_time_seconds"] = (
            sum(record["training_time_seconds"] for record in iteration_records)
            / len(iteration_records)
        )

    summary = {
        "format_version": SUMMARY_FORMAT_VERSION,
        "subset": "center150",
        "num_samples": len(completed_records),
        "eval_iterations": eval_iterations,
        "averages": averages,
        "samples": completed_records,
    }
    atomic_write_json(results_root / "center150_metrics_summary.json", summary)

    lines = [
        "OmniScene center150 汇总（150 个样本等权平均）",
        "iteration  view_group      PSNR       SSIM       LPIPS     L1         train_time_seconds",
    ]
    for iteration in eval_iterations:
        record = averages[str(iteration)]
        for group_name in (ALL_18_VIEW_GROUP, NOVEL_12_VIEW_GROUP):
            metrics = record["view_groups"][group_name]
            lines.append(
                f"{iteration:<10d} {group_name:<15s} {metrics['psnr']:<10.6f} "
                f"{metrics['ssim']:<10.6f} {metrics['lpips']:<10.6f} "
                f"{metrics['l1']:<10.6f} {record['training_time_seconds']:.3f}"
            )
    atomic_write_text(results_root / "center150_metrics_summary.txt", "\n".join(lines) + "\n")
    print(f"[Summary] {results_root / 'center150_metrics_summary.json'}")
    return True


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="运行 NexusGS OmniScene center150 里程碑实验；未知参数会透传给 train.py"
    )
    parser.add_argument("--dataset-root", default="datasets/omniscene")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument(
        "--mode",
        default="center150",
        choices=["center150", "train", "val", "test", "demo"],
    )
    parser.add_argument("--resolution", default="112x200")
    parser.add_argument("--valid-threshold", type=float, default=0.3)
    parser.add_argument("--bin-limit", type=int, default=None)
    parser.add_argument("--scene-indices", default=None, help="逗号分隔的 1-based 样本索引")
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument(
        "--eval-iterations", nargs="+", type=int, default=[1_000, 5_000, 10_000]
    )
    parser.add_argument("--densify-until-iter", type=int, default=None)
    parser.add_argument("--position-lr-max-steps", type=int, default=None)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="仅用已保存的 PNG 在 CPU 上补算/汇总指标，绝不启动训练",
    )
    parser.add_argument(
        "--metrics-cpu-threads", type=int, default=min(12, os.cpu_count() or 1)
    )
    parser.add_argument("--metrics-batch-size", type=int, default=4)
    args, train_overrides = parser.parse_known_args()
    if train_overrides and train_overrides[0] == "--":
        train_overrides = train_overrides[1:]
    return args, train_overrides


def main() -> None:
    args, train_overrides = parse_args()
    args.eval_iterations = sorted(set(args.eval_iterations))
    if not args.eval_iterations or args.eval_iterations[0] <= 0:
        raise ValueError("--eval-iterations 必须是正整数")
    if args.iterations <= 0 or args.eval_iterations[-1] != args.iterations:
        raise ValueError("--iterations 必须为正数，且等于最大的 --eval-iterations")
    args.densify_until_iter = (
        args.iterations if args.densify_until_iter is None else args.densify_until_iter
    )
    args.position_lr_max_steps = (
        args.iterations
        if args.position_lr_max_steps is None
        else args.position_lr_max_steps
    )
    validate_train_overrides(train_overrides)

    resolution = _parse_resolution(args.resolution)
    dataset = OmniSceneDataset(
        root=args.dataset_root, mode=args.mode, resolution=resolution
    )
    if args.mode == "center150" and len(dataset) != CENTER150_SAMPLE_COUNT:
        raise RuntimeError(f"center150 样本数不是 {CENTER150_SAMPLE_COUNT}: {len(dataset)}")

    if args.output_root is None:
        args.output_root = Path("output/omniscene_preprocessed")
        if args.mode == "center150":
            args.output_root /= "center150"
    if args.results_root is None:
        args.results_root = Path("output/omniscene_results")
        if args.mode == "center150":
            args.results_root /= "center150"
    output_root = args.output_root.resolve()
    results_root = args.results_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    configuration = experiment_configuration(
        args, dataset, resolution, train_overrides
    )
    experiment_fingerprint = ensure_experiment_configuration(
        results_root, configuration
    )
    indices = parse_scene_indices(args.scene_indices, len(dataset), args.bin_limit)

    if args.metrics_only:
        if train_overrides:
            raise ValueError("--metrics-only 不接受 train.py 透传参数")
        print(
            "[Metrics-only] 仅执行 CPU 指标回填与汇总；"
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
        )
        backfill_missing_view_groups(
            dataset,
            output_root,
            results_root,
            args.eval_iterations,
            experiment_fingerprint,
            args.metrics_cpu_threads,
            args.metrics_batch_size,
        )
        if args.mode == "center150":
            aggregate_center150(
                dataset,
                output_root,
                results_root,
                args.eval_iterations,
                experiment_fingerprint,
            )
        return

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    failures = []
    for index in indices:
        try:
            process_scene(
                args,
                dataset,
                index,
                resolution,
                output_root,
                results_root,
                experiment_fingerprint,
                train_overrides,
                env,
            )
        except Exception as error:
            if not args.keep_going:
                raise
            failures.append(
                {
                    "scene_index": index + 1,
                    "bin_token": dataset.bin_tokens[index],
                    "error": str(error),
                }
            )
            print(f"[Failed] {index + 1}/{len(dataset)}: {error}", file=sys.stderr)

    backfill_missing_view_groups(
        dataset,
        output_root,
        results_root,
        args.eval_iterations,
        experiment_fingerprint,
        args.metrics_cpu_threads,
        args.metrics_batch_size,
    )
    if args.mode == "center150":
        aggregate_center150(
            dataset,
            output_root,
            results_root,
            args.eval_iterations,
            experiment_fingerprint,
        )
    if failures:
        atomic_write_json(results_root / "failures.json", {"failures": failures})
        raise RuntimeError(f"本轮有 {len(failures)} 个样本失败，详见 failures.json")


if __name__ == "__main__":
    main()
