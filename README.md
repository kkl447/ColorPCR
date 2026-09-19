# ColorPCR：面向 HKU-MARS RGB 的彩色点云配准

本仓库是 [ColorPCR](https://openaccess.thecvf.com/content/CVPR2024/html/Mu_ColorPCR_Color_Point_Cloud_Registration_with_Multi-Stage_Geometric-Color_Fusion_CVPR_2024_paper.html) 的 HKU-MARS RGB 适配版本。模型沿用 ColorPCR 的多阶段几何—颜色融合结构，在 KPConv-FPN 的多个尺度注入 HSV 信息，并在 superpoint matching 阶段使用 GeoColor 结构嵌入。

当前公开入口仅包含 `experiments/ColorPCR` 下的 HKU-MARS RGB 训练、验证和测试流程。数据集、预训练权重及训练输出不包含在 Git 仓库中。

## 方法简介

ColorPCR 面向低重叠彩色点云配准，包含两个主要颜色增强模块：

- Hierarchical Color Enhanced Feature Extraction（CEFE）：在局部特征提取的多个尺度融合几何与颜色信息。
- GeoColor Superpoint Matching：在几何距离、角度关系之外编码颜色关系，用于提升 coarse-level patch correspondence 的可靠性。

整体流程为：

```text
XYZRGB point clouds
    -> RGB-to-HSV conversion
    -> color-enhanced KPConv-FPN
    -> GeoColor superpoint transformer
    -> coarse superpoint matching
    -> Sinkhorn fine matching
    -> Local-to-Global Registration
    -> rigid transformation
```

## 环境安装

推荐环境与原始 ColorPCR 保持一致：

- Ubuntu 20.04
- Python 3.8
- PyTorch 1.7.1
- CUDA 11.0/11.1
- Open3D 0.11.2

```bash
conda create -n colorpcr python=3.8
conda activate colorpcr

conda install pytorch=1.7.1 cudatoolkit=11.0 -c pytorch
pip install -r requirements.txt
conda install -c open3d-admin open3d=0.11.2

python setup.py build develop
```

`setup.py build develop` 会编译点云 grid subsampling 和 radius neighbor search 扩展，因此需要可用的 C++/CUDA 编译环境。

## 数据集

### 数据来源

本实验使用 [MARS-LVIG（HKU-MARS）](https://mars.hku.hk/dataset.html) 中带有同步 LiDAR、RGB 图像和位姿信息的数据。请按照数据集官网的许可和引用要求下载原始数据。

本仓库使用以下四个序列：

| 划分 | 序列 | 配准对数量 |
| --- | --- | ---: |
| train | `AMvalley03`、`HKairport03`、`HKisland03` | 942 |
| val | `AMtown03` | 593 |
| test | `AMtown03` | 1184 |

处理后的每帧点云保存为 `Nx6 float32` NumPy 数组：

```text
[x, y, z, r, g, b]
```

其中 RGB 已归一化至 `[0, 1]`。数据加载时在线转换为 HSV；模型输入特征仍为全 1，颜色通过显式 HSV 路径进入网络。

### 数据预处理

HKU-MARS RGB 预处理工具位于配套的 [GeoTransformer 仓库](https://github.com/kkl447/GeoTransformer)中：

- [`pre_data_v015_dynamic_crop.py`](https://github.com/kkl447/GeoTransformer/blob/master/data/HKU_MARS/pre_data_v015_dynamic_crop.py)：动态裁剪与点云规模控制。
- [`pre_data_v015_dynamic_crop_rgb.py`](https://github.com/kkl447/GeoTransformer/blob/master/data/HKU_MARS/pre_data_v015_dynamic_crop_rgb.py)：将 LiDAR 投影到同步图像并生成 XYZRGB 点云。
- [`get_pkl_v015_dynamic.py`](https://github.com/kkl447/GeoTransformer/blob/master/data/HKU_MARS/get_pkl_v015_dynamic.py)：生成训练、验证和测试配准对 metadata。

假设原始数据位于 `/path/to/interval5_CAM_LIDAR`，可执行：

```bash
git clone https://github.com/kkl447/GeoTransformer.git
cd GeoTransformer/data/HKU_MARS

HKU_RGB_PREPROCESS_PARSE_ARGS=1 \
python pre_data_v015_dynamic_crop_rgb.py \
  --raw-root /path/to/interval5_CAM_LIDAR \
  --output-root /path/to/MARS_Dataset_v015_dynamic_s030_rgb \
  --seqs AMtown03 AMvalley03 HKairport03 HKisland03

HKU_PKL_PARSE_ARGS=1 \
python get_pkl_v015_dynamic.py \
  --base-dir /path/to/MARS_Dataset_v015_dynamic_s030_rgb \
  --metadata-dir metadata_amtown_valtest
```

默认预处理参数为：预下采样体素 `0.15 m`、模型体素 `0.30 m`、单帧最多 `30000` 点。metadata 使用最小双向 overlap `0.20`，并保存相对位姿和点云相对路径。

### 必要目录结构

训练所需的最小数据目录如下；`ply/` 可用于可视化，但不是训练必需项。

```text
MARS_Dataset_v015_dynamic_s030_rgb/
├── downsampled/
│   ├── AMtown03/
│   │   ├── 000000.npy
│   │   └── ...
│   ├── AMvalley03/
│   ├── HKairport03/
│   └── HKisland03/
└── metadata_amtown_valtest/
    ├── train.pkl
    ├── val.pkl
    ├── test.pkl
    └── get_pkl_v015_dynamic_config.json
```

每个 metadata 条目至少包含：

```text
seq, frame0, frame1, pcd0, pcd1, transform, overlap
```

## 训练

训练入口为 `experiments/ColorPCR/trainval.py`。数据集通过环境变量指定，不要求放在本仓库内，也不要求与 GeoTransformer 仓库位于同一目录。

```bash
cd experiments/ColorPCR

CUDA_VISIBLE_DEVICES=0 \
HKU_RGB_DATASET_ROOT=/path/to/MARS_Dataset_v015_dynamic_s030_rgb \
HKU_METADATA_DIR=metadata_amtown_valtest \
python trainval.py
```

默认配置为单卡 batch size 1、梯度累积 5 次、训练 90 个 epoch。训练日志、TensorBoard events 和 checkpoint 默认写入 `output/ColorPCR/`。

多 GPU 训练可使用：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
HKU_RGB_DATASET_ROOT=/path/to/MARS_Dataset_v015_dynamic_s030_rgb \
HKU_METADATA_DIR=metadata_amtown_valtest \
python -m torch.distributed.launch \
  --nproc_per_node=2 \
  --master_port=29501 \
  trainval.py
```

## 验证与测试

预训练权重不存放在 Git 仓库中。下载或训练得到 checkpoint 后，将其路径传给 `test.py`：

```bash
cd experiments/ColorPCR

CUDA_VISIBLE_DEVICES=0 \
HKU_RGB_DATASET_ROOT=/path/to/MARS_Dataset_v015_dynamic_s030_rgb \
HKU_METADATA_DIR=metadata_amtown_valtest \
python test.py \
  --snapshot=../../weights/epoch-87.pth.tar \
  --benchmark=val

HKU_RGB_DATASET_ROOT=/path/to/MARS_Dataset_v015_dynamic_s030_rgb \
HKU_METADATA_DIR=metadata_amtown_valtest \
python eval_hku.py --benchmark=val
```

测试集只需将两处 `val` 替换为 `test`：

```bash
CUDA_VISIBLE_DEVICES=0 \
HKU_RGB_DATASET_ROOT=/path/to/MARS_Dataset_v015_dynamic_s030_rgb \
HKU_METADATA_DIR=metadata_amtown_valtest \
python test.py \
  --snapshot=../../weights/epoch-87.pth.tar \
  --benchmark=test

HKU_RGB_DATASET_ROOT=/path/to/MARS_Dataset_v015_dynamic_s030_rgb \
HKU_METADATA_DIR=metadata_amtown_valtest \
python eval_hku.py --benchmark=test
```

注册成功标准为 `RRE < 5 deg` 且 `RTE < 2 m`。

## 参考结果

以下结果使用 `epoch-87.pth.tar`、`metadata_amtown_valtest` 和上述成功标准：

| 划分 | RR | RRE | RTE | PIR | IR | RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| val | 1.000 | 0.491 | 0.663 | 0.841 | 0.457 | 0.385 |
| test | 1.000 | 0.507 | 0.724 | 0.841 | 0.464 | 0.368 |

由于训练包含随机采样和数据增强，不同硬件与软件版本下的数值可能略有波动。默认随机种子为 `7351`。

## 数据、权重与输出

- 原始或处理后的 HKU-MARS 数据不上传到本 Git 仓库。
- checkpoint 不进入 Git 历史；建议通过 GitHub Release 单独发布，并提供 SHA256。
- `output/`、日志、TensorBoard events、测试特征和注册结果均由 `.gitignore` 排除。

## 引用

如果本仓库对你的研究有帮助，请引用原始 ColorPCR 论文：

```bibtex
@inproceedings{mu2024colorpcr,
  title={ColorPCR: Color Point Cloud Registration with Multi-Stage Geometric-Color Fusion},
  author={Mu, Juncheng and Bie, Lin and Du, Shaoyi and Gao, Yue},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={21061--21070},
  year={2024}
}
```

同时请按照 MARS-LVIG 官方页面的要求引用数据集论文。

## 致谢

本项目建立在以下工作和代码基础上：

- [ColorPCR](https://github.com/mujc2021/ColorPCR)
- [GeoTransformer](https://github.com/qinzheng93/GeoTransformer)
- [PREDATOR](https://github.com/prs-eth/OverlapPredator)
- [CoFiNet](https://github.com/haoyu94/Coarse-to-fine-correspondences)
- [MARS-LVIG](https://mars.hku.hk/dataset.html)

## 许可证

代码遵循本仓库 [`LICENSE`](LICENSE) 中的 MIT License。数据集版权与许可归 MARS-LVIG 发布方所有，代码许可证不覆盖数据集。
