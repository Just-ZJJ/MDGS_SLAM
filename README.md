
# MDGS-SLAM: Real-Time RGB-D Gaussian SLAM with Multi-View Densification for Scene Reconstruction

<h3 align="center"><a href="">Paper</a> | <a href="https://www.bilibili.com/video/BV16sGXzFEms/?share_source=copy_web&vd_source=1bf14fa95c6af9b6f95ec59f828e77e9">Video</a> | <a href="">Project Page</a></h3>


<table>
    <tr>
        <td ><center><img src="https://github.com/Justzhongjj/MD_SLAM/blob/main/media/fr3_office.gif" ></center></td>
        <td ><center><img src="https://github.com/Justzhongjj/MD_SLAM/blob/main/media/room0.gif" ></center></td>
    </tr>
    <tr>
        <td><center><img src="https://github.com/Justzhongjj/MD_SLAM/blob/main/media/office0.gif" ></center></td>
        <td ><center><img src="https://github.com/Justzhongjj/MD_SLAM/blob/main/media/office3.gif"  ></center> </td>
    </tr>
    <tr>
        <td><center><img src="https://github.com/Justzhongjj/MD_SLAM/blob/main/media/balloon.gif" ></center></td>
        <td ><center><img src="https://github.com/Justzhongjj/MD_SLAM/blob/main/media/mydata.gif"  ></center> </td>
    </tr>
</table>


This repository contains the official authors implementation associated with the paper "MDGS-SLAM: Real-time 3D Reconstruction
at Scale Using Gaussian Splatting", which can be found [here](链接).

## 1. Installation

### 1.1 Clone the Repository

```
git clone --recursive https://github.com/Justzhongjj/MDGS_SLAM
```

### 1.2 Python Environment
RTG-SLAM has been tested on python 3.9, CUDA=11.6, pytorch=1.12.1. The simplest way to install all dependences is to use [anaconda](https://www.anaconda.com/) and [pip](https://pypi.org/project/pip/) in the following steps: 

```bash
conda env create -f environment.yaml
```

### 1.3 Modified ORB-SLAM2 Python Binding
We have made some changes on ORB-SLAM2 to work with our ICP front-end and you can run this script to install pangolin, opencv, orbslam and boost-python binding. 

```bash
bash build_orb.sh
```

If you encounted the problem during install pangolin:

```bash
xxx/Pangolin/src/video/drivers/ffmpeg.cpp: In function ‘std::__cxx11::string pangolin::FfmpegFmtToString(AVPixelFormat)’:
xxx/Pangolin/src/video/drivers/ffmpeg.cpp:41:41: error: ‘AV_PIX_FMT_XVMC_MPEG2_MC’ was not declared in this scope
```

You can follow this [solution](https://github.com/stevenlovegrove/Pangolin/pull/318/files?diff=split&w=0).

#### Note
For real data, backend optimization based on ORB-SLAM2 is crucial. Therefore, you need to install the python binding for ORB-SLAM2 according to the steps. We have modified some code based on [ORB_SLAM2-PythonBindings
](https://github.com/jskinn/ORB_SLAM2-PythonBindings). If you encounter any problem related to compilation, you can refer to [ORB_SLAM2-PythonBindings
](https://github.com/jskinn/ORB_SLAM2-PythonBindings) to find solutions. Our ICP front-end works well when the depth is accurate so if you only want to test on synthetic dataset like Replica, you don't need to install ORB-SLAM2 python binding.


### 1.4 Test ORB-SLAM2 Python Binding
```bash
cd thirdParty/pybind/examples
python orbslam_rgbd_tum.py # please set voc_path, association_path ...
python eval_ate.py path_to_groundtruth.txt trajectory.txt --plot PLOT --verbose
```
If you encount the problem `SIGSEGV` similar to  [raulmur/ORB_SLAM2#844](https://github.com/raulmur/ORB_SLAM2/pull/844)  [silencht/SG-SLAM#31](https://github.com/silencht/SG-SLAM/issues/31), you can try to modify 'thirdParty/ORB-SLAM2-PYBIND/CMakeLists.txt'
```
#set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -march=native")
#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
```

If the code runs without any error and the trajetory is corret, you can move on to the next step.  
If you encounter any issues while installing the ORB-SLAM2 Python Binding, you can refer to [RTG-SLAM](https://github.com/MisEty/RTG-SLAM).

## 2. Dataset Preparation
### 2.1 Replica
```
bash scripts/download_replica.sh
```
### 2.2 TUM-RGBD
```bash
bash scripts/download_tum.sh
```
And copy config file to data folder.
```bash
cp configs/tum/dataset/fr1_desk.yaml data/TUM_RGBD/rgbd_dataset_freiburg1_desk/config.yaml
cp configs/tum/dataset/fr2_xyz.yaml data/TUM_RGBD/rgbd_dataset_freiburg2_xyz/config.yaml
cp configs/tum/dataset/fr3_office.yaml data/TUM_RGBD/rgbd_dataset_freiburg3_long_office_household/config.yaml
```

### 2.4 Ours
You can donwload our dataset from [google cloud](https://drive.google.com/drive/folders/161QHjVTHRCED9WmRWAlOEhJQ_GXxgtn5?usp=sharing). 


### 2.5 Dataset

```
|-- data
    
    |-- Replica
        |-- office0
            |-- results
            |-- office0.ply
            |-- traj.txt
        |-- cam_params.json
    |-- TUM_RGBD
        |-- rgbd_dataset_freiburg1_desk
```

## 3. Run
### 3.1 Replica
```bash

# Multi Process:
python slam.py --config ./configs/replica/office0.yaml
```

### 3.2 TUM-RGBD
```bash

# Multi Process: 
python slam.py --config ./configs/tum/fr1_desk.yaml
```

### 3.4 Ours
```bash
# Single Process: 
python slam.py --config ./configs/ours/hotel.yaml
# Multi Process: 
python slam_mp.py --config ./configs/ours/hotel.yaml
```

## 4. Evaluate
To evaluate our method, please add --eval to the command line argument:
### 4.1 Replica
```bash
python slam.py --config ./configs/replica/office0.yaml --eval
```
### 4.2 TUM-RGBD
```bash
python slam_mp.py --config ./configs/tum/fr1_desk.yaml --eval
```
<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@article{peng2024rtgslam,
        author    = {Zhexi Peng and Tianjia Shao and Liu Yong and Jingke Zhou and Yin Yang and Jingdong Wang and Kun Zhou},
        title     = {RTG-SLAM: Real-time 3D Reconstruction at Scale using Gaussian Splatting},
        booktitle  = {ACM SIGGRAPH Conference Proceedings, Denver, CO, United States, July 28 - August 1, 2024},
        year      = {2024},
      }</code></pre>
  </div>
</section>


## Acknowledgments
This work incorporates many open-source codes. We extend our gratitude to the authors of the software.
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [Differential Gaussian Rasterization
](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [SIBR_viewers](https://gitlab.inria.fr/sibr/sibr_core)
- [Tiny Gaussian Splatting Viewer](https://github.com/limacv/GaussianSplattingViewer)
- [Open3D](https://github.com/isl-org/Open3D)
- [Point-SLAM](https://github.com/eriksandstroem/Point-SLAM)
- [MonoGS](https://github.com/muskie82/MonoGS)
- [RTG-SLAM](https://github.com/MisEty/RTG-SLAM)
- [MVGS](https://github.com/xiaobiaodu/MVGS)
    
We thank all the authors for their great work.
