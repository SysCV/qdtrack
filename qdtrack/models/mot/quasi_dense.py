import matplotlib
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import os
import random
import seaborn as sns
from matplotlib.patches import Rectangle
from mmdet.core import bbox2result
from mmdet.models import TwoStageDetector
from PIL import ImageColor

from qdtrack.core import track2result
from ..builder import MODELS, build_tracker

matplotlib.use('Agg')


def random_color(seed):
    random.seed(seed)
    colors = sns.color_palette(n_colors=64)
    color = random.choice(colors)
    return color


@MODELS.register_module()
class QuasiDenseFasterRCNN(TwoStageDetector):

    def __init__(self, tracker=None, *args, **kwargs):
        self.prepare_cfg(kwargs)
        super().__init__(*args, **kwargs)
        self.tracker_cfg = tracker

    def prepare_cfg(self, kwargs):
        if kwargs.get('train_cfg', False):
            kwargs['roi_head']['track_train_cfg'] = kwargs['train_cfg'].get(
                'embed', None)

    def init_tracker(self):
        self.tracker = build_tracker(self.tracker_cfg)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_match_indices,
                      ref_img,
                      ref_img_metas,
                      ref_gt_bboxes,
                      ref_gt_labels,
                      ref_gt_match_indices,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      ref_gt_bboxes_ignore=None,
                      ref_gt_masks=None,
                      **kwargs):
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
        rpn_losses, proposal_list = self.rpn_head.forward_train(
            x,
            img_metas,
            gt_bboxes,
            gt_labels=None,
            gt_bboxes_ignore=gt_bboxes_ignore,
            proposal_cfg=proposal_cfg)
        losses.update(rpn_losses)

        ref_x = self.extract_feat(ref_img)
        ref_proposals = self.rpn_head.simple_test_rpn(ref_x, ref_img_metas)

        roi_losses = self.roi_head.forward_train(
            x, img_metas, proposal_list, gt_bboxes, gt_labels,
            gt_match_indices, ref_x, ref_img_metas, ref_proposals,
            ref_gt_bboxes, ref_gt_labels, gt_bboxes_ignore, gt_masks,
            ref_gt_bboxes_ignore, **kwargs)
        losses.update(roi_losses)

        return losses

    def simple_test(self, img, img_metas, rescale=False):
        # TODO inherit from a base tracker
        assert self.roi_head.with_track, 'Track head must be implemented.'
        frame_id = img_metas[0].get('frame_id', -1)
        if frame_id == 0:
            self.init_tracker()

        x = self.extract_feat(img)
        proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        det_bboxes, det_labels, track_feats = self.roi_head.simple_test(
            x, img_metas, proposal_list, rescale)

        if track_feats is not None:
            bboxes, labels, ids = self.tracker.match(
                bboxes=det_bboxes,
                labels=det_labels,
                track_feats=track_feats,
                frame_id=frame_id)

        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.roi_head.bbox_head.num_classes)

        if track_feats is not None:
            track_result = track2result(bboxes, labels, ids)
        else:
            from collections import defaultdict
            track_result = defaultdict(list)
        return dict(bbox_result=bbox_result, track_result=track_result)

    def show_result(self,
                    img,
                    result,
                    thickness=1,
                    font_scale=0.5,
                    show=False,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        # bbox_result = result['bbox_result']
        # segm_result = result['segm_result']
        track_result = result['track_result']

        if isinstance(img, str):
            img = plt.imread(img)
        else:
            img = mmcv.bgr2rgb(img)
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.autoscale(False)
        plt.subplots_adjust(
            top=1, bottom=0, right=1, left=0, hspace=None, wspace=None)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        # draw segtrack masks
        for id, item in track_result.items():
            # mask = item['segm']
            # img[mask] = img[mask] * 0.5 + color * 0.5
            bbox = item['bbox']
            label = item['label']
            bbox_int = bbox.astype(np.int32)
            left_top = (bbox_int[0], bbox_int[1])
            w = bbox_int[2] - bbox_int[0] + 1
            h = bbox_int[3] - bbox_int[1] + 1
            color = random_color(id)
            plt.gca().add_patch(
                Rectangle(
                    left_top, w, h, thickness, edgecolor=color, facecolor='none'))
            label_text = '{}'.format(int(id))
            bg_height = 15
            bg_width = 12
            bg_width = len(label_text) * bg_width
            plt.gca().add_patch(
                Rectangle((left_top[0], left_top[1] - bg_height),
                        bg_width,
                        bg_height,
                        thickness,
                        edgecolor=color,
                        facecolor=color))
            plt.text(left_top[0] - 1, left_top[1], label_text, fontsize=5)

        dir_name, file_name = os.path.split(out_file)
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        if out_file is not None:
            plt.savefig(out_file, dpi=300, bbox_inches='tight', pad_inches=0.0)
        plt.show()
        plt.clf()

        return img
