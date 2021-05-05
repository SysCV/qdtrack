import mmcv
import numpy as np
import os
import random
from mmdet.core import bbox2result

from qdtrack.core import track2result
from ..builder import MODELS, build_tracker
from .quasi_dense import QuasiDenseFasterRCNN



@MODELS.register_module()
class QuasiDenseFasterRCNNMulti(QuasiDenseFasterRCNN):

    def forward(self, img, img_metas, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels,
                      gt_instance_ids, *args, **kwargs):
        losses = dict()

        img_list = img
        img_metas_list = img_metas
        gt_bboxes_list = gt_bboxes
        gt_labels_list = gt_labels
        gt_instance_ids_list = gt_instance_ids

        with torch.no_grad():
            x_list = [self.extract_feat(img) for img in img_list]
            proposals_list = [
                self.rpn_head.simple_test_rpn(x, img_metas)
                for x, img_metas in zip(x_list, img_metas_list)]

        roi_losses = self.roi_head.forward_train(
            x_list, img_metas_list, proposals_list, gt_bboxes_list,
            gt_labels_list, gt_instance_ids_list)
        losses.update(roi_losses)

        return losses

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.
        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary infomation.
        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                # dist.all_reduce(loss_value.div_(dist.get_world_size()))
                value_list = [loss_value.clone()
                              for _ in range(self.world_size)]
                dist.all_gather(value_list, loss_value)
                value_list = torch.stack(value_list)
                valid_sum = self.world_size - value_list.isnan().int().sum()
                loss_value = value_list.nansum() / valid_sum
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars
