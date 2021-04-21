import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.models import HEADS, build_loss

from qdtrack.core import cal_similarity

from .quasi_dense_embed_head import QuasiDenseEmbedHead


@HEADS.register_module()
class SparseMatchEmbedHead(QuasiDenseEmbedHead):

    def get_track_targets(self, gt_match_indices, ref_bboxes):
        targets, weights = [], []
        for indices, ref_gts in zip(gt_match_indices, ref_bboxes):
            ref_gt_inds = torch.arange(ref_gts.size(0)).to(indices.device)
            _targets = (indices.view(-1, 1) == ref_gt_inds.view(1, -1)).int()
            _weights = (_targets.max(dim=1)[0] > 0).float()
            targets.append(_targets)
            weights.append(_weights)
        return targets, weights

    def match(self, key_embeds, ref_embeds):
        dists, cos_dists = [], []
        for key_embed, ref_embed in zip(key_embeds, ref_embeds):
            dist = cal_similarity(
                key_embed,
                ref_embed,
                method='dot_product')
            dists.append(dist)
            if self.loss_track_aux is not None:
                cos_dist = cal_similarity(
                    key_embed, ref_embed, method='cosine')
                cos_dists.append(cos_dist)
            else:
                cos_dists.append(None)
        return dists, cos_dists
