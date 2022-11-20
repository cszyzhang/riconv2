# RIConv++: Effective Rotation Invariant Convolutions for 3D Point Clouds Deep Learning

International Journal of Computer Vision, 2022

[Zhiyuan Zhang](https://zhiyuanzhang.net/), [Binh-Son Hua](https://sonhua.github.io/), [Sai-Kit Yeung](https://www.saikit.org/).

[Project Page](https://zhiyuanzhang.net/projects/riconv) | [arXiv](https://arxiv.org/pdf/2202.13094.pdf)

If you found this paper useful in your research, please cite:
```

@article{zhang2022riconv2,
  title={RIConv++: Effective Rotation Invariant Convolutions for 3D Point Clouds Deep Learning},
  author={Zhang, Zhiyuan and Hua, Binh-Son and Yeung, Sai-Kit},
  journal={International Journal of Computer Vision},
  volume={130},
  number={5},
  pages={1228--1243},
  year={2022},
  publisher={Springer}
}

@inproceedings{zhang2020global,
  title={Global Context Aware Convolutions for 3D Point Cloud Understanding},
  author={Zhang, Zhiyuan and Hua, Binh-Son and Chen, Wei and Tian, Yibin and Yeung, Sai-Kit},
  booktitle={2020 International Conference on 3D Vision (3DV)},
  pages={210--219},
  year={2020}
}

@inproceedings{zhang2019rotation,
  title={Rotation Invariant Convolutions for 3D Point Clouds Deep Learning},
  author={Zhang, Zhiyuan and Hua, Binh-Son and Rosen, David W and Yeung, Sai-Kit},
  booktitle={2019 International Conference on 3D Vision (3DV)},
  pages={204--213},
  year={2019}
}
```

## Installation
This repo provides the RIConv++ source codes, which had been tested with Python 3.9.7, PyTorch 1.9.0, CUDA 11.1 on Ubuntu 20.04. Our codes are implemented based on Xu Yan's Pytorch implementation of [PointNet++(Pytorch)](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) and ShaoShuai Shi's [PointNet++ cuda operations](https://github.com/sshaoshuai/Pointnet2.PyTorch).  

Install the pointnet++ cuda operation library by running the following command:
```
cd models/pointnet2/
python setup.py install
cd ../../
```

## Usage
### Classification
We perform classification on ModelNet40 and ScanObjectNN respectively.
#### ModelNet40

Download alignment **ModelNet** [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and save in `../data/modelnet40_normal_resampled/`. Follow the instructions of [PointNet++(Pytorch)](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) to prepare the data. Specifically, please use `--process_data` to preprocess the data, and move the processed data to `../data/modelnet40_preprocessed/`. Alternatively, you can also download the pre-processd data [here](https://1drv.ms/u/s!AmHXm1tT3NIcnnBiRlVxATXtOhe9?e=oynmh2) and save it in `../data/modelnet40_preprocessed/`. (**Note**: the `data/` folder is outside the project folder)

To train a RIConv++ model to classify shapes in the ModelNet40 dataset:
```
python3 train_classification_modelnet40.py
```
For testing, you can use your trained model by specifying `--log_dir` or use our **pretrained model** [4.9MB](log/classification_modelnet40/pretrained) directly:
```
python3 test_classification_modelnet40.py
```
#### ScanObjectNN
Download the **ScanObjectNN** [here](https://hkust-vgd.github.io/scanobjectnn/) and save the `main_split` and `main_split_nobg` subfolders that inlcude the h5 files into the `../data/scanobjectnn/` (**Note**: the `data/` folder is outside the project folder)

Training on the **OBJ_ONLY** variant:
```
python3 train_classification_scanobj.py --data_type 'OBJ_NOBG'
```
For testing, you can use your trained model by specifying `--log_dir` or use our **pretrained model** [4.9MB](log/classification_scanobj/pretrained/OBJ_NOBG) directly:
```
python3 test_classification_scanobj.py --data_type 'OBJ_NOBG'
```

Training on the **OBJ_BG** variant:
```
python3 train_classification_scanobj.py --data_type 'OBJ_BG'
```
For testing, you can use your trained model by specifying `--log_dir` or use our **pretrained model** [4.9MB](log/classification_scanobj/pretrained/OBJ_BG) directly:
```
python3 test_classification_scanobj.py --data_type 'OBJ_BG'
```

Training on the hardest variant **PB_T50_RS**:
```
python3 train_classification_scanobj.py --data_type 'hardest'
```
For testing, you can use your trained model by specifying `--log_dir` or use our **pretrained model** [4.9MB](log/classification_scanobj/pretrained/hardest) directly:
```
python3 test_classification_scanobj.py --data_type 'hardest'
```

### Segmentation
We perform part segmentation and semantic segmentation on ShapeNet and S3DIS respectively.

#### ShapeNet
Download alignment **ShapeNet** [here](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip)  and save in `../data/shapenetcore_partanno_segmentation_benchmark_v0_normal/`. (**Note**: the `data/` folder is outside the project folder)

Training:
```
python3 train_partseg.py
```
For testing, you can use your trained model by specifying `--log_dir` or use our **pretrained model** [18.2MB](log/part_seg/pretrained) directly:
```
python3 test_partseg.py
```

#### S3DIS
Please download the **S3DIS** dataset [here](http://buildingparser.stanford.edu/dataset.html#Download), and run the following scripts to preprocess the data:
```
cd data_utils
python collect_indoor3d_data.py
```
Processed data will save in `../data/s3dis/stanford_indoor3d/`. (**Note**: the `data/` folder is outside the project folder)

Training:
```
python3 train_semseg.py
```
For testing, you can use your trained model by specifying `--log_dir` or use our **pretrained model** [18.2MB](log/sem_seg/pretrained) directly:
```
python3 test_semseg.py
```

## License
This repository is released under MIT License (see LICENSE file for details).
