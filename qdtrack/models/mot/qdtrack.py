import numpy as np
from mmdet.core import bbox2result
from mmdet.models import build_detector, build_head
from mmdet.models.detectors.base import BaseDetector

from qdtrack.core import imshow_tracks, restore_result, track2result
from ..builder import MODELS, build_tracker


@MODELS.register_module()
class QDTrack(BaseDetector):

    def __init__(self,
                 detector=None,
                 track_head=None,
                 tracker=None,
                 freeze_detector=False,
                 *args,
                 **kwargs):
        super().__init__()
        self.tracker_cfg = tracker

        if detector is not None:
            self.detector = build_detector(detector)

        if track_head is not None:
            self.track_head = build_head(track_head)

        self.freeze_detector = freeze_detector
        if self.freeze_detector:
            self._freeze_detector()

    def _freeze_detector(self):

        self.detector = [
            self.backbone, self.neck, self.rpn_head, self.roi_head.bbox_head
        ]
        for model in self.detector:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

    def init_tracker(self):
        self.tracker = build_tracker(self.tracker_cfg)

    @property
    def with_track_head(self):
        """bool: whether the framework has a track_head."""
        return hasattr(self, 'track_head') and self.track_head is not None

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
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      ref_gt_bboxes_ignore=None,
                      ref_gt_masks=None,
                      **kwargs):
        """Forward function during training.

         Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            gt_bboxes (list[Tensor]): Ground truth bboxes of the image,
                each item has a shape (num_gts, 4).
            gt_labels (list[Tensor]): Ground truth labels of all images.
                each has a shape (num_gts,).
            gt_match_indices (list(Tensor)): Mapping from gt_instance_ids to
                ref_gt_instance_ids of the same tracklet in a pair of images.
            ref_img (Tensor): of shape (N, C, H, W) encoding input reference
                images. Typically these should be mean centered and std scaled.
            ref_img_metas (list[dict]): list of reference image info dict where
                each dict has: 'img_shape', 'scale_factor', 'flip', and may
                also contain 'filename', 'ori_shape', 'pad_shape',
                and 'img_norm_cfg'.
            ref_gt_bboxes (list[Tensor]): Ground truth bboxes of the
                reference image, each item has a shape (num_gts, 4).
            ref_gt_labels (list[Tensor]): Ground truth labels of all
                reference images, each has a shape (num_gts,).
            gt_masks (list[Tensor]) : Masks for each bbox, has a shape
                (num_gts, h , w).
            gt_bboxes_ignore (list[Tensor], None): Ground truth bboxes to be
                ignored, each item has a shape (num_ignored_gts, 4).
            ref_gt_bboxes_ignore (list[Tensor], None): Ground truth bboxes
                of reference images to be ignored,
                each item has a shape (num_ignored_gts, 4).
            ref_gt_masks (list[Tensor]) : Masks for each reference bbox,
                has a shape (num_gts, h , w).

        Returns:
            dict[str : Tensor]: All losses.
        """
        x = self.detector.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.detector.with_rpn:
            proposal_cfg = self.detector.train_cfg.get(
                'rpn_proposal', self.detector.test_cfg.rpn)
            rpn_losses, proposal_list = self.detector.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)

        roi_losses = self.detector.roi_head.forward_train(
            x, img_metas, proposal_list, gt_bboxes, gt_labels,
            gt_bboxes_ignore, gt_masks, **kwargs)

        losses.update(roi_losses)

        ref_x = self.detector.extract_feat(ref_img)
        ref_proposals = self.detector.rpn_head.simple_test_rpn(
            ref_x, ref_img_metas)

        track_losses = self.track_head.forward_train(
            x, img_metas, proposal_list, gt_bboxes, gt_labels,
            gt_match_indices, ref_x, ref_img_metas, ref_proposals,
            ref_gt_bboxes, ref_gt_labels, gt_bboxes_ignore, gt_masks,
            ref_gt_bboxes_ignore)

        losses.update(track_losses)

        return losses

    def simple_test(self, img, img_metas, rescale=False):
        # TODO inherit from a base tracker
        assert self.with_track_head, 'Track head must be implemented.'
        frame_id = img_metas[0].get('frame_id', -1)
        if frame_id == 0 and hasattr(self,
                                     'tracker') is False:  # for param search
            self.init_tracker()

        x = self.detector.extract_feat(img)
        proposal_list = self.detector.rpn_head.simple_test_rpn(x, img_metas)

        det_bboxes, det_labels = self.detector.roi_head.simple_test_bboxes(
            x,
            img_metas,
            proposal_list,
            self.detector.roi_head.test_cfg,
            rescale=rescale)

        det_bboxes = det_bboxes[0]
        det_labels = det_labels[0]

        track_feats = self.track_head.extract_bbox_feats(
            x, det_bboxes, img_metas)

        if track_feats is not None:
            bboxes, labels, ids = self.tracker.match(
                bboxes=det_bboxes,
                labels=det_labels,
                track_feats=track_feats,
                frame_id=frame_id)

        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.detector.roi_head.bbox_head.num_classes)

        if track_feats is not None:
            track_result = track2result(
                bboxes, labels, ids,
                self.detector.roi_head.bbox_head.num_classes)
        else:
            track_result = [
                np.zeros((0, 6), dtype=np.float32)
                for i in range(self.detector.roi_head.bbox_head.num_classes)
            ]
        return dict(bbox_results=bbox_result, track_results=track_result)

    def show_result(self,
                    img,
                    result,
                    thickness=1,
                    font_scale=0.5,
                    show=False,
                    out_file=None,
                    wait_time=0,
                    backend='cv2',
                    **kwargs):
        """Visualize tracking results.

        Args:
            img (str | ndarray): Filename of loaded image.
            result (dict): Tracking result.
                The value of key 'track_results' is ndarray with shape (n, 6)
                in [id, tl_x, tl_y, br_x, br_y, score] format.
                The value of key 'bbox_results' is ndarray with shape (n, 5)
                in [tl_x, tl_y, br_x, br_y, score] format.
            thickness (int, optional): Thickness of lines. Defaults to 1.
            font_scale (float, optional): Font scales of texts. Defaults
                to 0.5.
            show (bool, optional): Whether show the visualizations on the
                fly. Defaults to False.
            out_file (str | None, optional): Output filename. Defaults to None.
            backend (str, optional): Backend to draw the bounding boxes,
                options are `cv2` and `plt`. Defaults to 'cv2'.

        Returns:
            ndarray: Visualized image.
        """
        assert isinstance(result, dict)
        track_result = result.get('track_results', None)
        bboxes, labels, ids = restore_result(track_result, return_ids=True)
        img = imshow_tracks(
            img,
            bboxes,
            labels,
            ids,
            classes=self.CLASSES,
            thickness=thickness,
            font_scale=font_scale,
            show=show,
            out_file=out_file,
            wait_time=wait_time,
            backend=backend)
        return img

    def extract_feat(self, imgs):
        return super().extract_feat(imgs)

    def aug_test(self, imgs, img_metas, **kwargs):
        return super().aug_test(imgs, img_metas, **kwargs)
