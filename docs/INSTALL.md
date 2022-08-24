
## Installation

### Requirements
- Linux
- Python 3.6+ 
- PyTorch 1.3+
- CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
- NCCL 2+
- GCC 5+
- [mmcv](https://github.com/open-mmlab/mmcv) = 1.6.1
- [mmdetection](https://github.com/open-mmlab/mmdetection) = 2.25.1

### Install QDTrack

a. Create a conda virtual environment and activate it.
```shell
conda create -n qdtrack python=3.7 -y
conda activate qdtrack
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
conda install pytorch torchvision -c pytorch
```

Note: Make sure that your compilation CUDA version and runtime CUDA version match.
You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

c. Install mmcv and mmdetection.

```shell
pip install mmcv-full==1.6.1
pip install mmdet==2.25.1
```

You can also refer to the [offical instructions](https://github.com/open-mmlab/mmdetection/blob/master/docs/install.md).

d. Install QDTrack
```shell
python setup.py develop
```

