import random

import mmcv
import numpy as np
from mmdet.datasets import DATASETS, CocoDataset

from qdtrack.core import eval_mot
from .parsers import CocoVID
from .coco_video_dataset import CocoVideoDataset


@DATASETS.register_module()
class EMCocoVideoDataset(CocoVideoDataset):

    def pre_img_sampling(self, img_info):
        if img_info.get('frame_id', -1) <= 0:
            ref_img_info = img_info.copy()
        else:
            vid_id = img_info['video_id']
            img_ids = self.coco.get_img_ids_from_vid(vid_id)
            frame_id = img_info['frame_id']
            ref_img_id = img_ids[frame_id - 1]
            ref_img_info = self.coco.loadImgs([ref_img_id])[0]
            ref_img_info['filename'] = ref_img_info['file_name']
        return ref_img_info

    def prepare_test_img(self, idx):
        """Get testing data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data and annotation after pipeline with new keys \
                introduced by pipeline.
        """
        img_info = self.data_infos[idx]
        pre_img_info = self.pre_img_sampling(img_info)

        results = dict(img_info=img_info)
        pre_results = dict(img_info=pre_img_info)

        self.pre_pipeline([results, pre_results])
        return self.pipeline([results, pre_results])