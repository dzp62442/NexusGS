import os
import os.path as osp
import json
import pickle as pkl
import re
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

__all__ = [
    "CENTER150_FILENAME",
    "CENTER150_SAMPLE_COUNT",
    "OmniSceneDataset",
    "load_center150_tokens",
    "load_conditions",
    "load_info",
]


CENTER150_FILENAME = "bins_center150_v1.json"
CENTER150_SAMPLE_COUNT = 150
BIN_TOKEN_PATTERN = re.compile(r"^scene([0-9a-f]+)_bin(\d+)$")


def _load_json_object(path: str, description: str) -> dict:
    if not osp.isfile(path):
        raise FileNotFoundError(f"{description}不存在: {path}")
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{description}必须是 JSON object: {path}")
    return data


def _parse_bin_token(bin_token: str) -> Tuple[str, int]:
    if not isinstance(bin_token, str):
        raise ValueError(f"bin token 必须是字符串，实际为: {type(bin_token).__name__}")
    match = BIN_TOKEN_PATTERN.fullmatch(bin_token)
    if match is None:
        raise ValueError(f"非法 bin token: {bin_token}")
    return match.group(1), int(match.group(2))


def _expected_center150_tokens(val_manifest: dict) -> List[str]:
    """按 SVF-GS 的 lower-median 规则在内存中复算官方 center150。"""
    all_bins = val_manifest.get("bins")
    adjacent_bins = val_manifest.get("adjacent_bins")
    if not isinstance(all_bins, list) or not isinstance(adjacent_bins, list):
        raise ValueError("bins_val_3.2m.json 必须包含 list 字段 'bins' 和 'adjacent_bins'")
    if len(adjacent_bins) != CENTER150_SAMPLE_COUNT:
        raise ValueError(
            f"OmniScene val 应包含 {CENTER150_SAMPLE_COUNT} 个场景，实际为 {len(adjacent_bins)}"
        )

    flattened_bins = [bin_token for scene_bins in adjacent_bins for bin_token in scene_bins]
    if flattened_bins != all_bins:
        raise ValueError("bins_val_3.2m.json 中 adjacent_bins 展平后与 bins 不完全一致")

    selected_bins = []
    selected_scenes = set()
    for scene_index, scene_bins in enumerate(adjacent_bins):
        if not isinstance(scene_bins, list) or not scene_bins:
            raise ValueError(f"val 场景分组 {scene_index} 为空或格式错误")
        parsed = [_parse_bin_token(bin_token) for bin_token in scene_bins]
        scene_tokens = {scene_token for scene_token, _ in parsed}
        bin_indices = [bin_index for _, bin_index in parsed]
        if len(scene_tokens) != 1:
            raise ValueError(f"val 场景分组 {scene_index} 混入了多个 scene token")
        if bin_indices != list(range(len(scene_bins))):
            raise ValueError(f"val 场景分组 {scene_index} 的 bin 序号不连续或未按序排列")

        scene_token = next(iter(scene_tokens))
        if scene_token in selected_scenes:
            raise ValueError(f"val 清单含重复 scene token: {scene_token}")
        selected_scenes.add(scene_token)
        selected_bins.append(scene_bins[(len(scene_bins) - 1) // 2])

    if len(selected_bins) != CENTER150_SAMPLE_COUNT or len(set(selected_bins)) != CENTER150_SAMPLE_COUNT:
        raise ValueError("从 val 清单复算后未得到 150 个唯一中央 bin")
    return selected_bins


def load_center150_tokens(version_dir: str) -> List[str]:
    """严格校验 SVF-GS 生成的 center150 清单后加载，不在本项目生成清单。"""
    val_path = osp.join(version_dir, "bins_val_3.2m.json")
    center150_path = osp.join(version_dir, CENTER150_FILENAME)
    val_manifest = _load_json_object(val_path, "OmniScene val 清单")
    center150_manifest = _load_json_object(center150_path, "SVF-GS center150 清单")

    actual_tokens = center150_manifest.get("bins")
    if not isinstance(actual_tokens, list):
        raise ValueError(f"{CENTER150_FILENAME} 必须包含 list 字段 'bins'")
    expected_tokens = _expected_center150_tokens(val_manifest)
    if actual_tokens != expected_tokens:
        mismatch_index = next(
            (idx for idx, pair in enumerate(zip(actual_tokens, expected_tokens)) if pair[0] != pair[1]),
            min(len(actual_tokens), len(expected_tokens)),
        )
        raise ValueError(
            f"{CENTER150_FILENAME} 与 SVF-GS lower-median 规则不一致，"
            f"首个差异索引为 {mismatch_index}"
        )

    bin_info_dir = osp.join(version_dir, "bin_infos_3.2m")
    missing_infos = [
        bin_token for bin_token in actual_tokens
        if not osp.isfile(osp.join(bin_info_dir, f"{bin_token}.pkl"))
    ]
    if missing_infos:
        preview = ", ".join(missing_infos[:3])
        raise FileNotFoundError(
            f"center150 有 {len(missing_infos)} 个 bin 缺少 bin info；前几个为: {preview}"
        )
    return list(actual_tokens)


def load_info(info: dict) -> Tuple[str, np.ndarray, np.ndarray]:
    """获取图像路径及 OpenCV 相机与关键帧 LiDAR 坐标系之间的外参."""
    img_path = info["data_path"]
    c2w = np.array(info["sensor2lidar_transform"], dtype=np.float32)
    if c2w.shape != (4, 4):
        raise ValueError(f"sensor2lidar_transform must have shape (4, 4), got {c2w.shape}")
    w2c = np.linalg.inv(c2w).astype(np.float32)
    return img_path, c2w, w2c


def _maybe_resize_image(img: Image.Image, target_hw: Tuple[int, int], ck: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    tgt_h, tgt_w = target_hw
    if img.height == tgt_h and img.width == tgt_w:
        return np.array(img), ck
    fx, fy = ck[0, 0], ck[1, 1]
    cx, cy = ck[0, 2], ck[1, 2]
    scale_h = tgt_h / img.height
    scale_w = tgt_w / img.width
    fx *= scale_w
    fy *= scale_h
    cx *= scale_w
    cy *= scale_h
    ck_resized = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    resized = img.resize((tgt_w, tgt_h), Image.BILINEAR)
    return np.array(resized), ck_resized


def load_conditions(img_paths: List[str], resolution: Tuple[int, int]):
    images = []
    intrins = []
    depths = []
    confs = []
    for path in img_paths:
        param_path = path.replace("samples", "samples_param_small")
        param_path = param_path.replace("sweeps", "sweeps_param_small")
        param_path = param_path.replace(".jpg", ".json")
        params = json.load(open(param_path))
        ck = np.array(params["camera_intrinsic"], dtype=np.float32)

        img_path = path.replace("samples", "samples_small")
        img_path = img_path.replace("sweeps", "sweeps_small")
        img = Image.open(img_path).convert("RGB")
        img_np, ck_resized = _maybe_resize_image(img, resolution, ck)
        images.append(img_np)

        norm_ck = ck_resized.copy()
        norm_ck[0, :] /= resolution[1]
        norm_ck[1, :] /= resolution[0]
        intrins.append(norm_ck.astype(np.float32))

        depth_path = img_path.replace("sweeps_small", "sweeps_dptm_small")
        depth_path = depth_path.replace("samples_small", "samples_dptm_small")
        depth_path = depth_path.replace(".jpg", "_dpt.npy")
        conf_path = depth_path.replace("_dpt.npy", "_conf.npy")
        depth = np.load(depth_path).astype(np.float32)
        conf = np.load(conf_path).astype(np.float32)
        if depth.shape != resolution:
            depth = np.array(Image.fromarray(depth).resize((resolution[1], resolution[0]), Image.BILINEAR))
        if conf.shape != resolution:
            conf = np.array(Image.fromarray(conf).resize((resolution[1], resolution[0]), Image.BILINEAR))
        depths.append(depth)
        confs.append(conf)

    images = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float() / 255.0
    intrins = torch.from_numpy(np.stack(intrins).astype(np.float32))
    depths = torch.from_numpy(np.stack(depths).astype(np.float32))
    confs = torch.from_numpy(np.stack(confs).astype(np.float32))
    return images, depths, confs, intrins


class OmniSceneDataset:
    """基础数据集，用于预处理阶段."""

    camera_types = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]

    def __init__(
        self,
        root: str,
        mode: str = "val",
        resolution: Tuple[int, int] = (112, 200),
        data_version: str = "interp_12Hz_trainval",
    ):
        self.data_root = root
        self.mode = mode
        self.resolution = resolution
        self.data_version = data_version
        version_dir = osp.join(self.data_root, self.data_version)

        if mode == "train":
            token_path = osp.join(version_dir, "bins_train_3.2m.json")
            self.bin_tokens = json.load(open(token_path))["bins"]
        elif mode == "val":
            token_path = osp.join(version_dir, "bins_val_3.2m.json")
            tokens = json.load(open(token_path))["bins"]
            self.bin_tokens = tokens[:30000:3000][:10]
        elif mode == "center150":
            self.bin_tokens = load_center150_tokens(version_dir)
        elif mode == "test":
            token_path = osp.join(version_dir, "bins_val_3.2m.json")
            tokens = json.load(open(token_path))["bins"]
            self.bin_tokens = tokens[0::14][:2048]
        elif mode == "demo":
            token_path = osp.join(version_dir, "bins_val_3.2m.json")
            self.bin_tokens = json.load(open(token_path))["bins"][:12]
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def __len__(self):
        return len(self.bin_tokens)

    def __getitem__(self, index: int):
        bin_token = self.bin_tokens[index]
        bin_path = osp.join(
            self.data_root,
            self.data_version,
            "bin_infos_3.2m",
            f"{bin_token}.pkl",
        )
        with open(bin_path, "rb") as f:
            bin_info = pkl.load(f)

        center_infos = {sensor: bin_info["sensor_info"][sensor][0] for sensor in self.camera_types}

        input_img_paths, input_c2ws, input_w2cs = [], [], []
        for cam in self.camera_types:
            info = center_infos[cam]
            img_path, c2w, w2c = load_info(info)
            img_path = img_path.replace("/datasets/nuScenes", self.data_root)
            input_img_paths.append(img_path)
            input_c2ws.append(c2w)
            input_w2cs.append(w2c)

        input_imgs, input_depths, input_confs, input_cks = load_conditions(input_img_paths, self.resolution)
        input_c2ws = torch.as_tensor(np.stack(input_c2ws), dtype=torch.float32)
        input_w2cs = torch.as_tensor(np.stack(input_w2cs), dtype=torch.float32)

        output_img_paths, output_c2ws, output_w2cs = [], [], []
        frame_num = len(bin_info["sensor_info"]["LIDAR_TOP"])
        assert frame_num >= 3, f"only got {frame_num} frames for bin {bin_token}"
        rend_indices = [[1, 2]] * len(self.camera_types)
        for cam_id, cam in enumerate(self.camera_types):
            for ind in rend_indices[cam_id]:
                info = bin_info["sensor_info"][cam][ind]
                img_path, c2w, w2c = load_info(info)
                img_path = img_path.replace("/datasets/nuScenes", self.data_root)
                output_img_paths.append(img_path)
                output_c2ws.append(c2w)
                output_w2cs.append(w2c)

        output_imgs, output_depths, output_confs, output_cks = load_conditions(output_img_paths, self.resolution)
        output_c2ws = torch.as_tensor(np.stack(output_c2ws), dtype=torch.float32)
        output_w2cs = torch.as_tensor(np.stack(output_w2cs), dtype=torch.float32)

        output_imgs = torch.cat([output_imgs, input_imgs], dim=0)
        output_depths = torch.cat([output_depths, input_depths], dim=0)
        output_confs = torch.cat([output_confs, input_confs], dim=0)
        output_c2ws = torch.cat([output_c2ws, input_c2ws], dim=0)
        output_w2cs = torch.cat([output_w2cs, input_w2cs], dim=0)
        output_cks = torch.cat([output_cks, input_cks], dim=0)

        return {
            "bin_token": bin_token,
            "context": {
                "images": input_imgs,
                "c2w": input_c2ws,
                "w2c": input_w2cs,
                "intrinsics": input_cks,
                "depths": input_depths,
                "confs": input_confs,
                "paths": input_img_paths,
            },
            "target": {
                "images": output_imgs,
                "c2w": output_c2ws,
                "w2c": output_w2cs,
                "intrinsics": output_cks,
                "depths": output_depths,
                "confs": output_confs,
                "paths": output_img_paths + input_img_paths,
            },
        }
