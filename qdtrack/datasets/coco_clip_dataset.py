import random

from numpy.core.numerictypes import maximum_sctype

import mmcv
import numpy as np
from mmdet.datasets import DATASETS

from qdtrack.core import eval_mot
from .coco_video_dataset import CocoVideoDataset
from .parsers import CocoVID


@DATASETS.register_module()
class CocoClipDataset(CocoVideoDataset):

    CLASSES = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.load_as_video

    def clip_sampling(self,
                      img_info,
                      interval,
                      clip_len=2):
        assert clip_len >= 2

        vid_id = img_info['video_id']
        img_ids = self.coco.get_img_ids_from_vid(vid_id)
        frame_id = img_info['frame_id']

        max_interval = (clip_len - 1) * interval
        left = max(0, frame_id - max_interval)
        right = min(frame_id + 1, len(img_ids) - max_interval)
        valid_inds = range(left, right)
        start_idx = random.choice(valid_inds)
        idxs = range(start_idx, start_idx + max_interval + 1, interval)
        img_ids = [img_ids[idx] for idx in idxs]

        if random.random() > 0.5:
            img_ids = list(reversed(img_ids))

        img_infos = self.coco.load_imgs(img_ids)
        for img_info in img_infos:
            img_info['filename'] = img_info['file_name']
        return img_infos

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """
        img_info = self.data_infos[idx]
        img_infos = self.clip_sampling(img_info, **self.ref_img_sampler)

        results_list = [
            self.prepare_results(img_info) for img_info in img_infos]

        self.pre_pipeline(results_list)
        return self.pipeline(results_list)
