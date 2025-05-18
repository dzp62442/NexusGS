import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from safetensors.torch import load_file
import numpy as np

class HFGaussianModel(
        nn.Module,
        PyTorchModelHubMixin,
        library_name="NexusGS",
        repo_url="https://github.com/USMizuki/NexusGS",
        paper_url="https://arxiv.org/abs/2503.18794",
        docs_url="https://usmizuki.github.io/NexusGS/",
        # ^ optional metadata to generate model card
    ):
    def __init__(self, n_gaussians, cameras_extent, train_cameras_config=[], test_cameras_config=[]):
        super().__init__()
        self.n_gaussians = n_gaussians
        self.cameras_extent = cameras_extent
        self.train_cameras_config = train_cameras_config
        self.test_cameras_config = test_cameras_config

        self.points = nn.Parameter(torch.zeros((n_gaussians, 3)).float().cuda())
        self.colors = nn.Parameter(torch.zeros((n_gaussians, 3)).float().cuda())

        if len(train_cameras_config) > 0:
            self.init_cameras_parameter(train_cameras_config, "train")
        if len(test_cameras_config) > 0:
            self.init_cameras_parameter(test_cameras_config, "test")

    def init_cameras_parameter(self, configs, prefix="train"):

        for idx, c in enumerate(configs):         

            setattr(self, prefix + "_cam{}_original_image".format(str(idx).zfill(3)), nn.Parameter(torch.zeros((3, c["image_height"], c["image_width"])).float().cuda()))
            setattr(self, prefix + "_cam{}_depth".format(str(idx).zfill(3)), nn.Parameter(torch.zeros((c["image_height"], c["image_width"])).float().cuda()))
            setattr(self, prefix + "_cam{}_depth_mask".format(str(idx).zfill(3)), nn.Parameter(torch.zeros((c["image_height"], c["image_width"])).float().cuda()))
            setattr(self, prefix + "_cam{}_alpha_mask".format(str(idx).zfill(3)), nn.Parameter(torch.ones((c["image_height"], c["image_width"])).float().cuda()))

    def load_from_gaussian_model(self, gaussians, scene):
        self.gaussians = gaussians

        self.points = nn.Parameter(torch.tensor(np.asarray(scene.pcd.points)).float().cuda())
        self.colors = nn.Parameter(torch.tensor(np.asarray(scene.pcd.colors)).float().cuda())
        
        self.train_cameras_config = self.load_cameras_config(scene.getTrainCameras())
        self.test_cameras_config = self.load_cameras_config(scene.getTestCameras())
        self.load_cams_parameter(scene.getTrainCameras(), prefix="train")
        self.load_cams_parameter(scene.getTestCameras(), prefix="test")

        self._hub_mixin_config = self.get_config()
    
    def load_cameras_config(self, cams):
        configs = []
        for cam in cams:
            config = {}
            config["colmap_id"] = cam.colmap_id
            config["image_name"] = cam.image_name
            config["image_height"] = cam.image_height
            config["image_width"] = cam.image_width
            config["R"] = cam.R.tolist()
            config["T"] = cam.T.tolist()
            config["K"] = cam.K.transpose().tolist()
            config["FoVx"] = cam.FoVx
            config["FoVy"] = cam.FoVy
            config["uid"] = cam.uid
            config["trans"] = cam.trans.tolist()
            config["scale"] = cam.scale
            config["bounds"] = cam.bounds.tolist()

            configs.append(config)
        
        return configs

    def load_cams_parameter(self, cams, prefix="train"):

        for idx,cam in enumerate(cams):
            setattr(self, prefix + "_cam{}_original_image".format(str(idx).zfill(3)), nn.Parameter(torch.tensor(cam.original_image).float().cuda()))
            if hasattr(cam, "flow_depth"):
                setattr(self, prefix + "_cam{}_depth".format(str(idx).zfill(3)), nn.Parameter(torch.tensor(cam.flow_depth).float().cuda()))
            if hasattr(cam, "flow_depth_mask"):
                setattr(self, prefix + "_cam{}_depth_mask".format(str(idx).zfill(3)), nn.Parameter(torch.tensor(cam.flow_depth_mask).float().cuda()))
            if cam.alpha_mask is not None:
                setattr(self, prefix + "_cam{}_alpha_mask".format(str(idx).zfill(3)), nn.Parameter(torch.tensor(cam.alpha_mask).float().cuda()))

    def get_config(self):
        return {
            "n_gaussians" : self.n_gaussians,
            "cameras_extent" : self.cameras_extent,
            "train_cameras_config": self.train_cameras_config,
            "train_cameras_config": self.test_cameras_config
        }

