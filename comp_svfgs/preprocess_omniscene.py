import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

from .dataset_omniscene import OmniSceneDataset


def _parse_resolution(res_str: str) -> Tuple[int, int]:
    if "x" not in res_str:
        raise ValueError("resolution 格式应为 HxW，例如 112x200")
    h, w = res_str.lower().split("x")
    return int(h), int(w)


def _save_images(tensor: torch.Tensor, original_paths: List[str], out_dir: Path) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = tensor.permute(0, 2, 3, 1).cpu().numpy()
    imgs = (np.clip(imgs, 0.0, 1.0) * 255.0).astype(np.uint8)
    file_names = []
    for idx, (img, src) in enumerate(zip(imgs, original_paths)):
        stem = Path(src).stem
        fname = f"{idx:03d}_{stem}.png"
        Image.fromarray(img).save(out_dir / fname)
        file_names.append(fname)
    return file_names


def _export_block(block: Dict, out_dir: Path, resolution: Tuple[int, int], valid_threshold: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    image_names = _save_images(block["images"], block["paths"], out_dir / "images")

    depth_metric = block["depths"].cpu().numpy()
    confs = block["confs"].cpu().numpy()
    depth_valid = (confs > valid_threshold).astype(np.uint8)
    np.save(out_dir / "depth_metric.npy", depth_metric)
    np.save(out_dir / "depth_valid.npy", depth_valid)

    intrinsics = block["intrinsics"].cpu().numpy()
    c2w = block["c2w"].cpu().numpy()
    w2c = block["w2c"].cpu().numpy()
    image_rel_paths = [str(Path("images") / n) for n in image_names]
    camera_entries = []
    for idx, fname in enumerate(image_names):
        camera_entries.append(
            {
                "image": image_rel_paths[idx],
                "image_name": Path(fname).stem,
                "depth_index": idx,
                "intrinsic": intrinsics[idx].tolist(),
                "c2w": c2w[idx].tolist(),
                "w2c": w2c[idx].tolist(),
                "width": resolution[1],
                "height": resolution[0],
            }
        )
    with open(out_dir / "cameras.json", "w", encoding="utf-8") as f:
        json.dump({"views": camera_entries}, f, indent=2)

    np.savez(
        out_dir / "cams.npz",
        image_paths=np.array(image_rel_paths),
        image_names=np.array([entry["image_name"] for entry in camera_entries]),
        intrinsics=intrinsics,
        c2w=c2w,
        w2c=w2c,
        width=resolution[1],
        height=resolution[0],
    )
    return {"num_views": len(image_names)}


def _load_meta(scene_dir: Path) -> Dict:
    meta_path = scene_dir / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return {}


def _write_meta(scene_dir: Path, meta: Dict):
    with open(scene_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def preprocess_scene(sample: Dict, scene_dir: Path, resolution: Tuple[int, int], mode: str, valid_threshold: float):
    scene_dir.mkdir(parents=True, exist_ok=True)
    context_info = _export_block(sample["context"], scene_dir / "context", resolution, valid_threshold)
    target_info = _export_block(sample["target"], scene_dir / "target", resolution, valid_threshold)

    meta = {
        "bin_token": sample["bin_token"],
        "mode": mode,
        "resolution": {"height": resolution[0], "width": resolution[1]},
        "num_context": context_info["num_views"],
        "num_target": target_info["num_views"],
        "valid_threshold": valid_threshold,
        "near": 0.1,
        "far": 1000.0,
    }
    _write_meta(scene_dir, meta)


def preprocess_dataset(dataset: OmniSceneDataset, output_root: Path, resolution: Tuple[int, int], valid_threshold: float, limit: int = None):
    output_root.mkdir(parents=True, exist_ok=True)
    count = len(dataset) if limit is None else min(limit, len(dataset))
    for idx in range(count):
        sample = dataset[idx]
        scene_name = f"{idx + 1:02d}_{sample['bin_token']}"
        scene_dir = output_root / scene_name

        meta = _load_meta(scene_dir)
        if meta and meta.get("bin_token") == sample["bin_token"] and meta.get("mode") == dataset.mode:
            res = meta.get("resolution", {})
            if res.get("height") == resolution[0] and res.get("width") == resolution[1] and meta.get("valid_threshold") == valid_threshold:
                continue

        preprocess_scene(sample, scene_dir, resolution, dataset.mode, valid_threshold)


def main():
    parser = argparse.ArgumentParser(description="预处理 OmniScene 数据集")
    parser.add_argument("--dataset-root", type=str, default="datasets/omniscene")
    parser.add_argument("--mode", type=str, default="val", choices=["train", "val", "test", "demo"])
    parser.add_argument("--resolution", type=str, default="112x200")
    parser.add_argument("--output-root", type=str, default="output/omniscene_preprocessed")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--valid-threshold", type=float, default=0.3)
    args = parser.parse_args()

    resolution = _parse_resolution(args.resolution)
    dataset = OmniSceneDataset(root=args.dataset_root, mode=args.mode, resolution=resolution)
    preprocess_dataset(dataset, Path(args.output_root), resolution, args.valid_threshold, args.limit)


if __name__ == "__main__":
    main()
