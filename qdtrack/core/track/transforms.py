from collections import defaultdict
import numpy as np
import torch

def track2result(bboxes, labels, ids, num_classes=None):
    if num_classes is not None:
        valid_inds = ids > -1
        bboxes = bboxes[valid_inds]
        labels = labels[valid_inds]
        ids = ids[valid_inds]

        if bboxes.shape[0] == 0:
            return [np.zeros(bboxes.shape) for i in range(num_classes)]
        else:
            if isinstance(bboxes, torch.Tensor):
                bboxes = bboxes.cpu().numpy()
                labels = labels.cpu().numpy()
                ids = ids.cpu().numpy()
            return [
                np.concatenate((ids[labels == i, None], bboxes[labels == i, :]),
                               axis=1) for i in range(num_classes)
            ]
    else:
        valid_inds = ids > -1
        bboxes = bboxes[valid_inds].cpu().numpy()
        labels = labels[valid_inds].cpu().numpy()
        ids = ids[valid_inds].cpu().numpy()

        outputs = defaultdict(list)
        for bbox, label, id in zip(bboxes, labels, ids):
            outputs[id] = dict(bbox=bbox, label=label)
        return outputs


