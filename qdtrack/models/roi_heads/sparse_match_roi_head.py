import torch
from mmdet.core import bbox2roi, build_assigner, build_sampler
from mmdet.models import HEADS, build_head, build_roi_extractor
from mmdet.models.roi_heads import StandardRoIHead

from .quasi_dense_roi_head import QuasiDenseRoIHead


@HEADS.register_module()
class SparseMatchRoIHead(QuasiDenseRoIHead):

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_match_indices,
                      ref_x,
                      ref_img_metas,
                      ref_proposals,
                      ref_gt_bboxes,
                      ref_gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      ref_gt_bboxes_ignore=None,
                      *args,
                      **kwargs):
        losses = super(QuasiDenseRoIHead, self).forward_train(
            x, img_metas, proposal_list, gt_bboxes, gt_labels, gt_bboxes_ignore, gt_masks,
            *args, **kwargs)

        if self.with_track:
            key_bboxes, key_labels = gt_bboxes, gt_labels
            key_feats = self._track_forward(x, key_bboxes)
            ref_bboxes = ref_gt_bboxes
            ref_feats = self._track_forward(ref_x, ref_bboxes)

            num_key_bboxes = [len(key_bbox) for key_bbox in key_bboxes]
            num_ref_bboxes = [len(ref_bbox) for ref_bbox in ref_bboxes]

            match_feats = self.track_head.match(
                key_feats, ref_feats, num_key_bboxes, num_ref_bboxes)
            asso_targets = self.track_head.get_track_targets(gt_match_indices,
                                                             ref_gt_bboxes)
            loss_track = self.track_head.loss(*match_feats, *asso_targets)

        losses.update(loss_track)
        return loss_track
