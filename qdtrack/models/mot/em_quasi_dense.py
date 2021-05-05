import math
import numpy as np
import os
import random

import mmcv
import torch
import torch.nn as nn
from torch.nn import functional as F
from mmdet.core import bbox2result
from mmcv.runner import auto_fp16, force_fp32
from PIL import ImageColor

from qdtrack.core import track2result
from ..builder import MODELS, build_tracker
from .quasi_dense import QuasiDenseFasterRCNN


@MODELS.register_module()
class EMQuasiDenseFasterRCNN(QuasiDenseFasterRCNN):

    def __init__(self, channels, proto_num, stage_num, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.channels = channels
        self.proto_num = proto_num
        self.stage_num = stage_num

        for i in range(5):
            protos = torch.Tensor(1, channels, proto_num)
            protos.normal_(0, math.sqrt(2. / proto_num))
            protos = self._l2norm(protos, dim=1)
            self.register_buffer('mu%d' % i, protos)

        self.conv_v = nn.Conv2d(channels, channels, 1)
        self.conv_1 = nn.Conv2d(channels, channels, 1)
        self.conv_2 = nn.Conv2d(channels, channels, 1)

    @force_fp32(apply_to=('inp', ))
    def _l1norm(self, inp, dim):
        return inp / (1e-6 + inp.sum(dim=dim, keepdim=True))

    @force_fp32(apply_to=('inp', ))
    def _l2norm(self, inp, dim):
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))

    @force_fp32(apply_to=('feat', 'mu'))
    def _em_iter(self, feat, mu, first=False):
        B, C, H, W = feat.size()

        x = feat.view(B, C, -1)                             # B * C * N
        feat_v = self.conv_v(feat)
        x_v = feat_v.view(B, C, -1)                         # B * C * N

        with torch.no_grad():
            if first:
                mu = mu.repeat(B, 1, 1)
            for i in range(self.stage_num):
                z = torch.einsum('bcn,bck->bnk', (x, mu))   # B * N * K
                z = F.softmax(z, dim=2)                     # B * N * K
                z = self._l1norm(z, dim=1)                  # B * N * K
                mu = torch.einsum('bcn,bnk->bck', (x, z))   # B * C * K
                mu = self._l2norm(mu, dim=1)  

        mu_v = torch.einsum('bcn,bnk->bck', (x_v, z))       # B * C * K
        return mu, mu_v

    @force_fp32(apply_to=('feat', 'mu'))
    def _prop(self, feat, mu):
        B, C, H, W = feat.size()
        x = feat.view(B, C, -1)                             # B * C * N
        z = torch.einsum('bcn,bck->bnk', (x, mu))           # B * N * K
        z = F.softmax(z, dim=2)                             # B * N * K
        return z

    def cal_weight(self, feat1, feat2):
        feat1 = self.conv_1(feat1)
        feat2 = self.conv_2(feat2)
        w1 = F.normalize(feat1, dim=1)
        w2 = F.normalize(feat2, dim=1)
        w = (w1 * w2).sum(dim=1, keepdim=True)
        return w

    def forward(self, img, img_metas, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def em(self, x, ref_x):
        ys = []
        mus = [self.mu0, self.mu1, self.mu2, self.mu3, self.mu4]
        for i, (y, ref_y) in enumerate(zip(x, ref_x)):
            B, C, H, W = y.size()
            key_mu_k, key_mu_v = self._em_iter(y, mus[i], first=True)
            ref_mu_k, ref_mu_v = self._em_iter(ref_y, key_mu_k)

            key_z = self._prop(y, key_mu_k)                 # B * N * K
            ref_z = self._prop(y, ref_mu_k)                 # B * N * K

            key_r = torch.einsum('bck,bnk->bcn', (key_mu_v, key_z))
            ref_r = torch.einsum('bck,bnk->bcn', (ref_mu_v, ref_z))

            key_r = key_r.view(B, C, H, W)
            ref_r = ref_r.view(B, C, H, W)
            weight1 = self.cal_weight(key_r, key_r)
            weight2 = self.cal_weight(key_r, ref_r)

            weight = torch.cat((weight1, weight2), dim=1)
            weight = F.softmax(weight * 10, dim=1)
            weight1, weight2 = torch.split(weight, 1, dim=1)

            y = key_r * weight1 + ref_r * weight2
            ys.append(y)

            if self.training:
                mu = key_mu_k.mean(dim=0, keepdim=True)
                mus[i] = mus[i] * 0.9 + mu * 0.1

        return ys

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
        ref_x = self.extract_feat(ref_img)
        x = self.em(x, ref_x)

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

        ref_proposals = self.rpn_head.simple_test_rpn(ref_x, ref_img_metas)

        roi_losses = self.roi_head.forward_train(
            x, img_metas, proposal_list, gt_bboxes, gt_labels,
            gt_match_indices, ref_x, ref_img_metas, ref_proposals,
            ref_gt_bboxes, ref_gt_labels, gt_bboxes_ignore, gt_masks,
            ref_gt_bboxes_ignore, **kwargs)
        losses.update(roi_losses)

        return losses

    def forward_test(self, img, img_metas, ref_img, ref_img_metas, rescale=False):
        # TODO inherit from a base tracker
        assert self.roi_head.with_track, 'Track head must be implemented.'
        frame_id = img_metas[0].get('frame_id', -1)
        if frame_id == 0:
            self.init_tracker()

        x = self.extract_feat(img)
        ref_x = self.extract_feat(ref_img)
        x = self.em(x, ref_x)

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
