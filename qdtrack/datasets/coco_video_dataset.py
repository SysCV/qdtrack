import mmcv
import numpy as np
import os
import os.path as osp
import pycocotools.mask as mask_utils
import random
from functools import partial
from mmdet.datasets import DATASETS, CocoDataset
from multiprocessing import Pool
from PIL import Image
from scalabel.label.io import save
from scalabel.label.transforms import bbox_to_box2d
from scalabel.label.typing import Frame, Label
from tqdm import tqdm

from ..core import eval_mot
from ..core.evaluation import xyxy2xywh
from .parsers import CocoVID

CATEGORIES = [
    '', 'pedestrian', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
    'bicycle', 'traffic light', 'traffic sign']
SHAPE = [720, 1280]


@DATASETS.register_module()
class CocoVideoDataset(CocoDataset):

    CLASSES = None

    def __init__(self,
                 load_as_video=True,
                 match_gts=True,
                 skip_nomatch_pairs=True,
                 key_img_sampler=dict(interval=1),
                 ref_img_sampler=dict(
                     scope=3, num_ref_imgs=1, method='uniform'),
                 *args,
                 **kwargs):
        self.load_as_video = load_as_video
        self.match_gts = match_gts
        self.skip_nomatch_pairs = skip_nomatch_pairs
        self.key_img_sampler = key_img_sampler
        self.ref_img_sampler = ref_img_sampler
        super().__init__(*args, **kwargs)

    def load_annotations(self, ann_file):
        """Load annotation from annotation file."""
        if not self.load_as_video:
            data_infos = super().load_annotations(ann_file)
        else:
            data_infos = self.load_video_anns(ann_file)
        return data_infos

    def load_video_anns(self, ann_file):
        self.coco = CocoVID(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

        data_infos = []
        self.vid_ids = self.coco.get_vid_ids()
        self.img_ids = []
        for vid_id in self.vid_ids:
            img_ids = self.coco.get_img_ids_from_vid(vid_id)
            img_ids = self.key_img_sampling(img_ids, **self.key_img_sampler)
            self.img_ids.extend(img_ids)
            for img_id in img_ids:
                info = self.coco.load_imgs([img_id])[0]
                info['filename'] = info['file_name']
                data_infos.append(info)
        return data_infos

    def key_img_sampling(self, img_ids, interval=1):
        return img_ids[::interval]

    def ref_img_sampling(self,
                         img_info,
                         scope,
                         num_ref_imgs=1,
                         method='uniform'):
        if num_ref_imgs != 1 or method != 'uniform':
            raise NotImplementedError
        if img_info.get('frame_id', -1) < 0 or scope <= 0:
            ref_img_info = img_info.copy()
        else:
            vid_id = img_info['video_id']
            img_ids = self.coco.get_img_ids_from_vid(vid_id)
            frame_id = img_info['frame_id']
            if method == 'uniform':
                left = max(0, frame_id - scope)
                right = min(frame_id + scope, len(img_ids) - 1)
                valid_inds = img_ids[left:frame_id] + img_ids[frame_id +
                                                              1:right + 1]
                ref_img_id = random.choice(valid_inds)
            ref_img_info = self.coco.loadImgs([ref_img_id])[0]
            ref_img_info['filename'] = ref_img_info['file_name']
        return ref_img_info

    def _pre_pipeline(self, _results):
        super().pre_pipeline(_results)
        _results['frame_id'] = _results['img_info'].get('frame_id', -1)

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        if isinstance(results, list):
            for _results in results:
                self._pre_pipeline(_results)
        elif isinstance(results, dict):
            self._pre_pipeline(results)
        else:
            raise TypeError('input must be a list or a dict')

    def get_ann_info(self, img_info):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        img_id = img_info['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(img_info, ann_info)

    def prepare_results(self, img_info):
        ann_info = self.get_ann_info(img_info)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            idx = self.img_ids.index(img_info['id'])
            results['proposals'] = self.proposals[idx]
        return results

    def match_results(self, results, ref_results):
        match_indices, ref_match_indices = self._match_gts(
            results['ann_info'], ref_results['ann_info'])
        results['ann_info']['match_indices'] = match_indices
        ref_results['ann_info']['match_indices'] = ref_match_indices
        return results, ref_results

    def _match_gts(self, ann, ref_ann):
        if ann.get('instance_ids', False):
            ins_ids = list(ann['instance_ids'])
            ref_ins_ids = list(ref_ann['instance_ids'])
            match_indices = np.array([
                ref_ins_ids.index(i) if i in ref_ins_ids else -1
                for i in ins_ids
            ])
            ref_match_indices = np.array([
                ins_ids.index(i) if i in ins_ids else -1 for i in ref_ins_ids
            ])
        else:
            match_indices = np.arange(ann['bboxes'].shape[0], dtype=np.int64)
            ref_match_indices = match_indices.copy()
        return match_indices, ref_match_indices

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """
        img_info = self.data_infos[idx]
        ref_img_info = self.ref_img_sampling(img_info, **self.ref_img_sampler)

        results = self.prepare_results(img_info)
        ref_results = self.prepare_results(ref_img_info)

        if self.match_gts:
            results, ref_results = self.match_results(results, ref_results)
            nomatch = (results['ann_info']['match_indices'] == -1).all()
            if self.skip_nomatch_pairs and nomatch:
                return None

        self.pre_pipeline([results, ref_results])
        return self.pipeline([results, ref_results])

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_instance_ids = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                if ann.get('segmentation', False):
                    gt_masks_ann.append(ann['segmentation'])
                instance_id = ann.get('instance_id', None)
                if instance_id is not None:
                    gt_instance_ids.append(ann['instance_id'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        if self.load_as_video:
            ann['instance_ids'] = gt_instance_ids

        return ann

    def format_track_results(self, results, **kwargs):
        pass

    def evaluate(self,
                 results,
                 metric=['bbox', 'track'],
                 logger=None,
                 classwise=False,
                 mot_class_average=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=None,
                 metric_items=None):
        # evaluate for detectors without tracker
        eval_results = dict()
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'track']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        super_metrics = ['bbox', 'segm']
        super_metrics = [_ for _ in metrics if _ in super_metrics]
        if super_metrics:
            if 'bbox' in super_metrics and 'segm' in super_metrics:
                super_results = []
                for bbox, segm in zip(results['bbox_result'],
                                      results['segm_result']):
                    super_results.append((bbox, segm))
            else:
                super_results = results['bbox_result']
            super_eval_results = super().evaluate(
                results=super_results,
                metric=super_metrics,
                logger=logger,
                classwise=classwise,
                proposal_nums=proposal_nums,
                iou_thrs=iou_thr,
                metric_items=metric_items)
            eval_results.update(super_eval_results)

        if 'track' in metrics:
            track_eval_results = eval_mot(
                mmcv.load(self.ann_file),
                results['track_result'],
                class_average=mot_class_average)
            eval_results.update(track_eval_results)

        return eval_results

    def mask_prepare(self, track_dict):
        scores, colors, masks = [], [], []
        for id_, instance in track_dict.items():
            masks.append(mask_utils.decode(instance['segm']))
            colors.append([instance['label'] + 1, 0, id_ >> 8, id_ & 255])
            scores.append(instance['bbox'][-1])
        return scores, colors, masks

    def mask_merge(self, mask_infor, img_name, bitmask_base):
        scores, colors, masks = mask_infor
        bitmask = np.zeros((*SHAPE, 4), dtype=np.uint8)
        sorted_idxs = np.argsort(scores)
        for idx in sorted_idxs:
            for i in range(4):
                bitmask[..., i] = (
                    bitmask[..., i] * (1 - masks[idx]) +
                    masks[idx] * colors[idx][i])
        bitmask_path = osp.join(bitmask_base, img_name.replace('.jpg', '.png'))
        bitmask_dir = osp.split(bitmask_path)[0]
        if not osp.exists(bitmask_dir):
            os.makedirs(bitmask_dir)
        bitmask = Image.fromarray(bitmask)
        bitmask.save(bitmask_path)

    def mask_merge_parallel(self, track_dicts, img_names, bitmask_base, nproc):
        with Pool(nproc) as pool:
            print('\nCollecting mask information')
            mask_infors = pool.map(self.mask_prepare, tqdm(track_dicts))
            print('\nMerging overlapped masks.')
            pool.starmap(
                partial(self.mask_merge, bitmask_base=bitmask_base),
                tqdm(zip(mask_infors, img_names), total=len(mask_infors)))

    def det_to_bdd(self, results, out_base, nproc):
        bdd100k = []
        ann_id = 0
        print(f'\nStart converting to BDD100K detection format')
        for idx, bboxes_list in tqdm(enumerate(results['bbox_result'])):
            img_name = self.data_infos[idx]['file_name']
            frame = Frame(name=img_name, labels=[])

            for cls_, bboxes in enumerate(bboxes_list):
                for bbox in bboxes:
                    ann_id += 1
                    label = Label(
                        id=ann_id,
                        score=bbox[-1],
                        box2d=bbox_to_box2d(xyxy2xywh(bbox)),
                        category=CATEGORIES[cls_ + 1])
                    frame.labels.append(label)
            bdd100k.append(frame)

        print(f'\nWriting the converted json')
        out_path = osp.join(out_base, "det.json")
        save(out_path, bdd100k)

    def ins_seg_to_bdd(self, results, out_base, nproc=4):
        bdd100k = []
        bitmask_base = osp.join(out_base, "ins_seg")
        if not osp.exists(bitmask_base):
            os.makedirs(bitmask_base)

        track_dicts = []
        img_names = [
            self.data_infos[idx]['file_name']
            for idx in range(len(results['bbox_result']))]

        print(f'\nStart converting to BDD100K instance segmentation format')
        ann_id = 0
        for idx, [bboxes_list, segms_list] in enumerate(
                zip(results['bbox_result'], results['segm_result'])):
            index = 0
            frame = Frame(name=img_names[idx], labels=[])
            track_dict = {}
            for cls_, (bboxes, segms) in enumerate(zip(bboxes_list,
                                                       segms_list)):
                for bbox, segm in zip(bboxes, segms):
                    ann_id += 1
                    index += 1
                    label = Label(id=str(ann_id), index=index, score=bbox[-1])
                    frame.labels.append(label)
                    instance = {'bbox': bbox, 'segm': segm, 'label': cls_}
                    track_dict[index] = instance
            
            bdd100k.append(frame)
            track_dicts.append(track_dict)

        print(f'\nWriting the converted json')
        out_path = osp.join(out_base, 'ins_seg.json')
        save(out_path, bdd100k)

        self.mask_merge_parallel(track_dicts, img_names, bitmask_base, nproc)

    def box_track_to_bdd(self, results, out_base, nproc):
        bdd100k = []
        track_base = osp.join(out_base, "box_track")
        if not osp.exists(track_base):
            os.makedirs(track_base)

        print(f'\nStart converting to BDD100K box tracking format')
        for idx, track_dict in enumerate(results['track_result']):
            img_name = self.data_infos[idx]['file_name']
            frame = Frame(name=img_name, labels=[])

            for id_, instance in track_dict.items():
                bbox = instance['bbox']
                cls_ = instance['label']
                label = Label(
                    id=id_,
                    score=bbox[-1],
                    box2d=bbox_to_box2d(xyxy2xywh(bbox)),
                    category=CATEGORIES[cls_ + 1])
                frame.labels.append(label)
            bdd100k.append(frame)

        print(f'\nWriting the converted json')
        out_path = osp.join(out_base, "box_track.json")
        save(out_path, bdd100k)

    def seg_track_to_bdd(self, results, out_base, nproc=4):
        bitmask_base = osp.join(out_base, "seg_track")
        if not osp.exists(bitmask_base):
            os.makedirs(bitmask_base)

        print(f'\nStart converting to BDD100K seg tracking format')
        img_names = [
            self.data_infos[idx]['file_name']
            for idx in range(len(results['track_result']))]
        self.mask_merge_parallel(results['track_result'], img_names,
                                 bitmask_base, nproc)

    def preds2bdd100k(self, results, tasks, out_base, *args, **kwargs):
        metric2func = dict(
            det=self.det_to_bdd,
            ins_seg=self.ins_seg_to_bdd,
            box_track=self.box_track_to_bdd,
            seg_track=self.seg_track_to_bdd)

        for task in tasks:
            metric2func[task](results, out_base, *args, **kwargs)
