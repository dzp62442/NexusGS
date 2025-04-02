<div align="center">

<h1>NexusGS: Sparse View Synthesis with Epipolar Depth Priors in 3D Gaussian Splatting</h1>

<div>
    Yulong Zheng<sup>1</sup>&emsp;
    Zicheng Jiang<sup>1</sup>&emsp;
    Shengfeng He<sup>2</sup>&emsp;
    Yandu Sun<sup>1</sup>&emsp;
    Junyu Dong<sup>1</sup>&emsp;
    Huaidong Zhang<sup>3</sup>&emsp;
    Yong Du<sup>1, *</sup>
</div>

<div>
    <sup>1</sup>Ocean University of Chinay&emsp;
    <sup>2</sup>Singapore Management University&emsp;
    <sup>3</sup>South China University of Technology
</div>

<div>
    <sup>*</sup>corresponding author
</div>

[Paper](https://arxiv.org/abs/2503.18794) | 
[Project Page](https://usmizuki.github.io/NexusGS/) |
[Video](https://www.youtube.com/watch?v=K2foTIXzpMQ)

</div>


---------------------------------------------------
<p align="center" >
  <a href="">
    <img src="https://usmizuki.github.io/NexusGS/static/images/pipeline.png" alt="pipeline" width="100%">
  </a>
</p>


## Environmental Setups
Tested on Ubuntu 18.04, CUDA 11.8, PyTorch 2.0.0

```bash
conda env create -n nexus python=3.10
conda activate nexus
```

Install Pytorch

```bash
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```

Install submodules

```bash
pip install submodules/diff-gaussian-rasterization-confidence
pip install submodules/simple-knn
```

## Running

Taking LLFF as an example, the dataset folder structure is as follows:
```Shell
├── datasets
    ├── LLFF
        ├── scene
            ├── sparse
            ├── images 
            ├── images_8
            ├── 3_views
                ├── flow  
```

### LLFF

Download LLFF dataset: [Link](https://drive.google.com/drive/folders/1cK3UDIJqKAAm7zyrxRYVFJ0BRMgrwhh4).

Download the LLFF optical flow processed by [FlowFormer++](https://github.com/XiaoyuShi97/FlowFormerPlusPlus) from the [Link](https://drive.google.com/file/d/1DPr02s2wMznlNN0-qDOZRZohzmnnuanX/view?usp=drive_link).

Run using the following script:

```bash
sh scripts/run_llff.sh 0
```

### DTU

```
TODO
```

### MipNeRF-360

```
TODO
```

## Citation
If you find our work useful for your project, please consider citing the following paper.


``` bibtex
@article{zheng2025nexusgs,
  title={NexusGS: Sparse View Synthesis with Epipolar Depth Priors in 3D Gaussian Splatting},
  author={Zheng, Yulong and Jiang, Zicheng and He, Shengfeng and Sun, Yandu and Dong, Junyu and Zhang, Huaidong and Du, Yong},
  journal={arXiv preprint arXiv:2503.18794},
  year={2025}
}
```

## Acknowledgement

Special thanks to the following awesome projects!

- [Gaussian-Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [DreamGaussian](https://github.com/ashawkey/diff-gaussian-rasterization)
- [SparseNeRF](https://github.com/Wanggcong/SparseNeRF)
- [MipNeRF-360](https://github.com/google-research/multinerf)
- [FSGS](https://zehaozhu.github.io/FSGS/)
- [FlowFormer++](https://github.com/XiaoyuShi97/FlowFormerPlusPlus)