# FemtoDet
Official codes of ICCV2023 paper: <<Femtodet: an object detection baseline for energy versus performance tradeoffs>>

# Preparation
1. Download the dataset.
   We mainly train FemtoDet on [Pascal VOC 0712](http://host.robots.ox.ac.uk/pascal/VOC/), you should firstly download the datasets. By default, we assume the dataset is stored in ./data/.

2. Dataset preparation.
   Then, you can move all images to ./data/voc2coco/jpeg/*, and use our converted coco format [annotation files](https://pan.baidu.com/s/1SLgZd_2cLhLFC54lLM3sHg?pwd=umbz) (umbz).
   
# References
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
