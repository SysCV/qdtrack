import torch
from mmdet.core import bbox2roi, build_assigner, build_sampler
from mmdet.models import HEADS, build_head, build_roi_extractor
from mmdet.models.roi_heads import StandardRoIHead
import tensorflow as tf
import numpy as np
@HEADS.register_module()
class QuasiDenseRoIHead(StandardRoIHead):

    def __init__(self,
                 track_roi_extractor=None,
                 track_head=None,
                 track_train_cfg=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if track_head is not None:
            self.track_train_cfg = track_train_cfg
            self.init_track_head(track_roi_extractor, track_head)
            if self.track_train_cfg:
                self.init_track_assigner_sampler()
        self._signitures = {
            'image_files': 'image_files:0',
            'image_arrays': 'image_arrays:0',
            'prediction': 'detections:0',
        }
        self.efficient_det_path = '/home/erdos/workspace/pylot/dependencies/models/obstacle_detection/efficientdet/efficientdet-d7x/efficientdet-d7x_frozen.pb'
        self._model_name, self._tf_session = self.load_serving_model('efficientdet-d7x', self.efficient_det_path)

    def load_serving_model(self, model_name, model_path):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            # Load a frozen graph.
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as f:
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
        gpu_options = tf.GPUOptions(
            allow_growth=True,
            visible_device_list=str(0))
        return model_name, tf.Session(
            graph=detection_graph,
            config=tf.ConfigProto(gpu_options=gpu_options))

    def efficient_det_predict(self, inputs, device):
        img = inputs[0].permute(1, 2, 0).cpu().numpy()
        outputs_np = self._tf_session.run(
            self._signitures['prediction'],
            feed_dict={self._signitures['image_arrays']: [img]})[0]
        # _, ymin, xmin, ymax, xmax, score, _class
        outputs_np = outputs_np[~(outputs_np[:, 5]==0)]
        bboxes = outputs_np[:, 1:6]
        labels = outputs_np[:, 6]
        
        return [torch.from_numpy(bboxes).to(device)], [torch.from_numpy(labels).to(device)]

    def init_track_assigner_sampler(self):
        """Initialize assigner and sampler."""
        if self.track_train_cfg.get('assigner', None):
            self.track_roi_assigner = build_assigner(
                self.track_train_cfg.assigner)
            self.track_share_assigner = False
        else:
            self.track_roi_assigner = self.bbox_assigner
            self.track_share_assigner = True

        if self.track_train_cfg.get('sampler', None):
            self.track_roi_sampler = build_sampler(
                self.track_train_cfg.sampler, context=self)
            self.track_share_sampler = False
        else:
            self.track_roi_sampler = self.bbox_sampler
            self.track_share_sampler = True

    @property
    def with_track(self):
        """bool: whether the RoI head contains a `track_head`"""
        return hasattr(self, 'track_head') and self.track_head is not None

    def init_track_head(self, track_roi_extractor, track_head):
        """Initialize ``track_head``"""
        if track_roi_extractor is not None:
            self.track_roi_extractor = build_roi_extractor(track_roi_extractor)
            self.track_share_extractor = False
        else:
            self.track_share_extractor = True
            self.track_roi_extractor = self.bbox_roi_extractor
        self.track_head = build_head(track_head)

    def init_weights(self, *args, **kwargs):
        super().init_weights(*args, **kwargs)
        if self.with_track:
            self.track_head.init_weights()
            if not self.track_share_extractor:
                self.track_roi_extractor.init_weights()

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
        losses = super().forward_train(x, img_metas, proposal_list, gt_bboxes,
                                       gt_labels, gt_bboxes_ignore, gt_masks,
                                       *args, **kwargs)
        if self.with_track:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            if ref_gt_bboxes_ignore is None:
                ref_gt_bboxes_ignore = [None for _ in range(num_imgs)]
            key_sampling_results, ref_sampling_results = [], []
            for i in range(num_imgs):
                assign_result = self.track_roi_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.track_roi_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                key_sampling_results.append(sampling_result)

                ref_assign_result = self.track_roi_assigner.assign(
                    ref_proposals[i], ref_gt_bboxes[i], ref_gt_bboxes_ignore[i],
                    ref_gt_labels[i])
                ref_sampling_result = self.track_roi_sampler.sample(
                    ref_assign_result,
                    ref_proposals[i],
                    ref_gt_bboxes[i],
                    ref_gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in ref_x])
                ref_sampling_results.append(ref_sampling_result)

            key_bboxes = [res.pos_bboxes for res in key_sampling_results]
            key_feats = self._track_forward(x, key_bboxes)
            ref_bboxes = [res.bboxes for res in ref_sampling_results]
            ref_feats = self._track_forward(ref_x, ref_bboxes)

            match_feats = self.track_head.match(key_feats, ref_feats,
                                                key_sampling_results,
                                                ref_sampling_results)
            asso_targets = self.track_head.get_track_targets(
                gt_match_indices, key_sampling_results, ref_sampling_results)
            loss_track = self.track_head.loss(*match_feats, *asso_targets)

            losses.update(loss_track)

        return losses

    def _track_forward(self, x, bboxes):
        """Track head forward function used in both training and testing."""
        rois = bbox2roi(bboxes)
        track_feats = self.track_roi_extractor(
            x[:self.track_roi_extractor.num_inputs], rois)
        track_feats = self.track_head(track_feats)
        return track_feats

    def simple_test(self, x, img, img_metas, proposal_list, rescale):
        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        det_bboxes, det_labels = self.efficient_det_predict(img, det_bboxes[0].device)

        # TODO: support batch inference
        det_bboxes = det_bboxes[0]
        det_labels = det_labels[0]

        if det_bboxes.size(0) == 0:
            return det_bboxes, det_labels, None

        track_bboxes = det_bboxes[:, :-1] * torch.tensor(
            img_metas[0]['scale_factor']).to(det_bboxes.device)
        track_feats = self._track_forward(x, [track_bboxes])

        return det_bboxes, det_labels, track_feats
