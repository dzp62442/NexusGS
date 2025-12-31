import os
import os.path as osp
import json
import pickle as pkl
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

__all__ = ["OmniSceneDataset", "load_conditions", "load_info"]


def load_info(info: dict) -> Tuple[str, np.ndarray, np.ndarray]:
    """获取图像路径与外参."""
    img_path = info["data_path"]
    c2w = np.array(info["sensor2lidar_transform"], dtype=np.float32)
    lidar2cam_r = np.linalg.inv(info["sensor2lidar_rotation"])
    lidar2cam_t = info["sensor2lidar_translation"] @ lidar2cam_r.T
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = lidar2cam_r.T
    w2c[3, :3] = -lidar2cam_t
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

        if mode == "train":
            token_path = osp.join(self.data_root, self.data_version, "bins_train_3.2m.json")
            self.bin_tokens = json.load(open(token_path))["bins"]
        elif mode == "val":
            token_path = osp.join(self.data_root, self.data_version, "bins_val_3.2m.json")
            tokens = json.load(open(token_path))["bins"]
            self.bin_tokens = tokens[:30000:3000][:10]
        elif mode == "test":
            token_path = osp.join(self.data_root, self.data_version, "bins_val_3.2m.json")
            tokens = json.load(open(token_path))["bins"]
            self.bin_tokens = tokens[0::14][:2048]
        elif mode == "demo":
            token_path = osp.join(self.data_root, self.data_version, "bins_val_3.2m.json")
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
