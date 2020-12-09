# Quasi-Dense Tracking

This is the offical implementation of paper [Quasi-Dense Similarity Learning for Multiple Object Tracking](https://arxiv.org/pdf/2006.06664.pdf).

We present a [trailer](https://youtu.be/o8HRJAOZidc) that consists of method illustrations and tracking visualizations. Take a look!

If you have any questions, please go to [Discussions](https://github.com/SysCV/qdtrack/discussions).

## Abstract

Similarity learning has been recognized as a crucial step for object tracking. However, existing multiple object tracking methods only use sparse ground truth matching as the training objective, while ignoring the majority of the informative regions on the images. In this paper, we present Quasi-Dense Similarity Learning, which densely samples hundreds of region proposals on a pair of images for contrastive learning. We can naturally combine this similarity learning with existing detection methods to build Quasi-Dense Tracking (QDTrack) without turning to displacement regression or motion priors. We also find that the resulting distinctive feature space admits a simple nearest neighbor search at the inference time. Despite its simplicity, QDTrack outperforms all existing methods on MOT, BDD100K, Waymo, and TAO tracking benchmarks. It achieves 68.7 MOTA at 20.3 FPS on MOT17 without using external training data. Compared to methods with similar detectors, it boosts almost 10 points of MOTA and significantly decreases the number of ID switches on BDD100K and Waymo datasets. 



## Quasi-dense matching
![teaser](figures/teaser.png)

## Main results
With out bells and whistles, our method outperforms the states of the art on BDD100K and Waymo Tracking datasets by a large margin.

### Joint object detection and tracking on BDD100K test set

| mMOTA | mIDF1  | ID Sw. |
|-------|--------|--------|
| 35.2  | 51.8   |  11019 |



### Joint object detection and tracking on Waymo validation set

| Category   | MOTA | IDF1 | ID Sw. |
|------------|------|------|--------|
| Vehicle    | 55.6 | 66.2 | 24309  | 
| Pedestrian | 50.3 | 58.4 | 6347   |
| Cyclist    | 26.2 | 45.7 | 56     | 
| All        | 44.0 | 56.8 | 30712  | 


## Installation

Please refer to [INSTALL.md](docs/INSTALL.md) for installation instructions.


## Usages
Please refer to [GET_STARTED.md](docs/GET_STARTED.md) for dataset preparation and running instructions.

We propose [pretrained models](https://drive.google.com/file/d/1YNAQgd8rMqqEG-fRj3VWlO4G5kdwJbxz/view?usp=sharing) on BDD100K dataset as reference.

More models will be released later.


## Citation 

```
@article{qdtrack,
  title={Quasi-Dense Similarity Learning for Multiple Object Tracking},
  author={Pang, Jiangmiao and Qiu, Linlu and Li, Xia and Chen, Haofeng and Li, Qi and Darrell, Trevor and Yu, Fisher},
  journal={arXiv preprint arXiv:2006.06664},
  year={2020}
}
```
