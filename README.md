# Quasi-Dense Instance Similarity Learning

This is the offical implementation of paper [Quasi-Dense Instance Similarity learning](https://arxiv.org/pdf/2006.06664.pdf).

We present a [trailer](https://youtu.be/o8HRJAOZidc) that consists of method illustrations and tracking visualizations. Take a look!

If you have any questions or discussions, feel free to contact Jiangmiao Pang ([pangjiangmiao@gmail.com](mailto:pangjiangmiao@gmail.com)).

## Abstract

Similarity metrics for instances have drawn much attention, due to their importance for computer vision problems such as object tracking. However, existing methods regard object similarity learning as a post-hoc stage after object detection and only use sparse ground truth matching as the training objective. This process ignores the majority of the regions on the images. In this paper, we present a simple yet effective quasi-dense matching method to learn instance similarity from hundreds of region proposals in a pair of images. In the resulting feature space, a simple nearest neighbor search can distinguish different instances without bells and whistles. When applied to joint object detection and tracking, our method can outperform existing methods without using location or motion heuristics, yielding almost 10 points higher MOTA on BDD100K and Waymo tracking datasets. Our method is also competitive on one-shot object detection, which further shows the effectiveness of quasi-dense matching for category-level metric learning. 



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
