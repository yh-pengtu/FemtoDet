# FemtoDet
Official codes of ICCV2023 paper: <<Femtodet: an object detection baseline for energy versus performance tradeoffs>>

### Dependencies
* Python 3.8
* Torch 1.9.1+cu111
* Torchvision 0.10.1+cu111
* mmcv-full 1.4.2
* mmdet 2.23.0

### Installation
Do it as [mmdetection](https://github.com/open-mmlab/mmdetection/tree/v2.23.0) had done.

### Preparation
1. Download the dataset.
   
   We mainly train FemtoDet on [Pascal VOC 0712](http://host.robots.ox.ac.uk/pascal/VOC/), you should firstly download the datasets. By default, we assume the dataset is stored in ./data/.

2. Dataset preparation.
   
   Then, you can move all images to ./data/voc2coco/jpeg/*;you can use our converted coco format [annotation files](https://pan.baidu.com/s/1SLgZd_2cLhLFC54lLM3sHg?pwd=umbz)(umbz) and put these files to ./data/voc2coco/annotations/*; finally, the directory structure is

```
*data/voc2coco
    *jpeg
        *2008_003841.jpg
        *...
    *annotations
        *trainvoc_annotations.json
        *testvoc_annotations.json
```

3. Download the initialized models.
   We trained our designed backbone on ImageNet 1k, and used it for [the inite weights](https://pan.baidu.com/s/1DhrT675Va2wcPAi5aUc-bg?pwd=6tns)(6tns) of FemtoDet.

```
FemtoDet/weights/*
```

### Training
```
bash ./tools/train_femtodet.sh 4
```

### Results and Models

```
|  Detector  | Params | box AP50 |                  Config                          |   Download     |
------------------------------------------------------------------------------------------------------
|  FemtoDet  | 68.77k |          | [config](./configs/femtoDet/femtodet_0stage.py)  |                |
```

### References
If you find the code useful for your research, please consider citing:
```bib
@misc{tu2023femtodet,
      title={FemtoDet: An Object Detection Baseline for Energy Versus Performance Tradeoffs}, 
      author={Peng Tu and Xu Xie and Guo AI and Yuexiang Li and Yawen Huang and Yefeng Zheng},
      year={2023},
      eprint={2301.06719},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
