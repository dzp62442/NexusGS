import argparse
import json
import os,sys
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from comp_svfgs.dataset_omniscene import OmniSceneDataset
from comp_svfgs.preprocess_omniscene import preprocess_scene, _parse_resolution  # type: ignore


def _load_meta(scene_dir: Path):
    meta_path = scene_dir / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return {}


def ensure_scene_preprocessed(sample: dict, scene_dir: Path, resolution, valid_threshold: float, mode: str):
    meta = _load_meta(scene_dir)
    if meta:
        res = meta.get("resolution", {})
        if (
            meta.get("bin_token") == sample["bin_token"]
            and meta.get("mode") == mode
            and res.get("height") == resolution[0]
            and res.get("width") == resolution[1]
            and meta.get("valid_threshold") == valid_threshold
        ):
            return
    preprocess_scene(sample, scene_dir, resolution, mode, valid_threshold)


def run_command(cmd, env):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def main():
    parser = argparse.ArgumentParser(description="运行 NexusGS 的 OmniScene 实验")
    parser.add_argument("--dataset-root", type=str, default="datasets/omniscene")
    parser.add_argument("--output-root", type=str, default="output/omniscene_preprocessed")
    parser.add_argument("--results-root", type=str, default="output/omniscene_results")
    parser.add_argument("--mode", type=str, default="val", choices=["train", "val", "test", "demo"])
    parser.add_argument("--resolution", type=str, default="112x200")
    parser.add_argument("--valid-threshold", type=float, default=0.3)
    parser.add_argument("--bin-limit", type=int, default=None)
    parser.add_argument("--scene-indices", type=str, default=None, help="逗号分隔的索引（1-based）")
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--skip-metrics", action="store_true")
    args = parser.parse_args()

    resolution = _parse_resolution(args.resolution)
    dataset = OmniSceneDataset(root=args.dataset_root, mode=args.mode, resolution=resolution)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    if args.scene_indices:
        indices = [int(x) - 1 for x in args.scene_indices.split(",") if x.strip()]
    else:
        total = len(dataset) if args.bin_limit is None else min(args.bin_limit, len(dataset))
        indices = list(range(total))

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpus

    for idx in indices:
        sample = dataset[idx]
        scene_name = f"{idx + 1:02d}_{sample['bin_token']}"
        scene_dir = output_root / scene_name
        ensure_scene_preprocessed(sample, scene_dir, resolution, args.valid_threshold, args.mode)

        model_dir = results_root / scene_name
        model_dir.mkdir(parents=True, exist_ok=True)

        train_cmd = [
            "python",
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
            str(args.iterations),
            "--position_lr_max_steps",
            str(args.iterations),
            "--save_iterations",
            str(args.iterations),
            "--eval",
        ]
        run_command(train_cmd, env)

        render_cmd = [
            "python",
            "render.py",
            "--source_path",
            str(scene_dir),
            "--model_path",
            str(model_dir),
            "--dataset_type",
            "omniscene",
            "--iteration",
            str(args.iterations),
            "--render_depth",
        ]
        if not args.skip_render:
            run_command(render_cmd, env)

        metrics_cmd = [
            "python",
            "metrics.py",
            "--source_paths",
            str(scene_dir),
            "--model_paths",
            str(model_dir),
            "--iteration",
            str(args.iterations),
        ]
        if not args.skip_metrics:
            run_command(metrics_cmd, env)


if __name__ == "__main__":
    main()
