from mmdet.datasets import DATASETS

from .coco_video_dataset import CocoVideoDataset
from .coco_clip_dataset import CocoClipDataset
from .em_coco_video_dataset import EMCocoVideoDataset


@DATASETS.register_module()
class BDDVideoDataset(CocoVideoDataset):

    CLASSES = ('pedestrian', 'rider', 'car', 'bus', 'truck', 'bicycle',
               'motorcycle', 'train')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@DATASETS.register_module()
class BDDClipDataset(CocoClipDataset):

    CLASSES = ('pedestrian', 'rider', 'car', 'bus', 'truck', 'bicycle',
               'motorcycle', 'train')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@DATASETS.register_module()
class EMBDDVideoDataset(EMCocoVideoDataset):

    CLASSES = ('pedestrian', 'rider', 'car', 'bus', 'truck', 'bicycle',
               'motorcycle', 'train')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
