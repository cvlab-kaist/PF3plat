<p align="center">
  <h1 align="center">PF3plat: Pose-Free Feed-Forward 3D Gaussian Splatting</h1>
  <p align="center">
    <a href="https://sunghwanhong.github.io/">Sunghwan Hong<sup>*</sup></a>
    ·
    <a href="https://crepejung00.github.io/">Jaewoo Jung<sup>*</sup></a>
    ·
    <a href="https://scholar.google.com/citations?user=zu-I2fYAAAAJ&hl=en">Heeseong Shin</a>
    ·
    <a href="https://onground-korea.github.io/">Jisang Han</a>
    ·
    <a href="https://jlyang.org/">Jiaolong Yang<sup>†</sup></a>
    ·
    <a href="https://www.microsoft.com/en-us/research/people/cluo/">Chong Luo<sup>†</sup></a>
    ·
    <a href="https://cvlab.kaist.ac.kr/">Seungryong Kim<sup>†</sup></a>
  </p>
  <h3 align="center"><a href="https://arxiv.org/pdf/2410.22128">Paper </a> | <a href="https://cvlab-kaist.github.io/PF3plat">Project Page </a> </h3>
  <div align="center"></div>
</p>

<p align="center">
  <a href="">
    <img src="https://cvlab-kaist.github.io/PF3plat/static/images/overall-architecture.png" alt="Logo" width="100%">
  </a>
</p>

<p align="center">
<strong>PF3plat enables estimates multi-view consistent depth, accurate camera pose, and photorealistic images from uncalibrated image collections.</strong>
</p>



**What to expect:**

- 🛠️ [x] Training and evaluation code & scripts
- 🌍 [] Demo code, taking only RGB images, for an easy use 
- ⚡ [] Leveraging more recently released monocular metric depth estimation model, <a href="https://machinelearning.apple.com/research/depth-pro">DepthPro</a>  or  <a href="https://wangrc.site/MoGePage/">MoGe</a> (Check out Jiaolong's new paper!).  
- 🚀 [] Releasing more generalized model (trained on full set of DL3DV and RealEstate10K)


## Installation

Our code is developed based on pytorch 2.0.1, CUDA 12.1 and python 3.10. 

We recommend using [conda](https://docs.anaconda.com/miniconda/) for installation:

```bash
conda create -n pf3plat python=3.10
conda activate pf3plat

pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Preparing RealEstate10K, ACID and DL3DV Datasets

### Training Dataset
- For training on RealEstate10K and ACID, we primarily follow [pixelSplat](https://github.com/dcharatan/pixelsplat) and [MVSplat](https://github.com/donydchen/mvsplat) to train on 256x256 resolution.

- Please refer to [here](https://github.com/dcharatan/pixelsplat?tab=readme-ov-file#acquiring-datasets) for acquiring the processed 360p dataset (360x640 resolution).

- For DL3DV, please download from [here](https://drive.google.com/file/d/1QBjMoH1MimoUdu23OrsO1fUTn6Q5GlXO/view?usp=sharing). We include both training and evaluation sets in there. 

- Note that we use the subset of DL3DV for training, so if you want to train and evaluate in your own way, you can prepare the dataset by following the instructions [here](https://github.com/cvg/depthsplat/blob/main/DATASETS.md).
### Evaluation Dataset

- For evaluation on RealEstate10K and ACID, we  follow [CoPoNeRF](https://github.com/cvlab-kaist/CoPoNeRF) and
 use different evaluation splits. Please download from [Re10k](https://drive.google.com/file/d/1PRx3Mj9IJ3eGwg2ZN-8ZXYzjObbhfwjf/view?usp=sharing) and   [ACID](https://drive.google.com/file/d/16Ql2sESqYFfc9qOjdkElQOW_qKoMNvaH/view?usp=sharing).

- Note that the evaluation set, which contains 140 scenes for evaluation, is included along with the training set. 





## Training
We  observed that enabling flash attention leads to frequent NaN values. In this codebase, we set flash=False. We thus set the current batch size as 3. We trained our model with A6000. 

Note that for evaluation you need to specify the path to the datasets in config/experiment/{re10k,acid,dl3dv}.yaml or simply pass as argument in the command line. 


```
python -m src.main +experiment={re10k, acid, dl3dv} data_loader.train.batch_size=3
```


## Evaluation

The pretrained weights can be found [here](https://drive.google.com/file/d/1ylrN8HNcnt2VdHkFnRBvgHIr9hBoJG1c/view?usp=sharing). Note that for evaluation you need to specify the path to the datasets in config/experiment/{re10k,acid,dl3dv}.yaml or simply pass as argument in the command line. 

```
python -m src.main +experiment={re10k, acid}_test checkpointing.load=$PATHTOCKPT$ dataset/view_sampler=evaluation mode=test test.compute_score=true

python -m src.main +experiment=dl3dv_test checkpointing.load=$PATHTPCKPT$ dataset/view_sampler=evaluation mode=test test.compute_scores=true dataset.view_sampler.index_path=assets/evaluation_index_dl3dv_{5, 10}view.json
```

## Camera Conventions

The camera intrinsic matrices are normalized (the first row is divided by image width, and the second row is divided by image height).

The camera extrinsic matrices are OpenCV-style camera-to-world matrices ( +X right, +Y down, +Z camera looks into the screen).



## Citation

```
@article{hong2024pf3plat,
      title   = {PF3plat: Pose-Free Feed-Forward 3D Gaussian Splatting},
      author  = {Sunghwan Hong and Jaewoo Jung and Heeseong Shin and Jisang Han and Jiaolong Yang and Chong Luo and Seungryong Kim},
      journal = {arXiv preprint arXiv:2410.22128},
      year    = {2024}
    }
```



## Acknowledgements

We thank the following repos for their codes, which were used in our implementation: [pixelSplat](https://github.com/dcharatan/pixelsplat), [MVSplat](https://github.com/donydchen/mvsplat), [UniDepth v2](https://github.com/lpiccinelli-eth/UniDepth) and [LightGlue](https://github.com/cvg/LightGlue). We thank the original authors for their excellent work. I also thank [Haofei Xu](https://haofeixu.github.io/) for helping me making this repo.

