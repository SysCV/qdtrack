import os
import os.path as osp
from functools import partial
from multiprocessing import Pool

import numpy as np
import pycocotools.mask as mask_utils
from PIL import Image
from tqdm import tqdm

SHAPE = [720, 1280]


def mask_prepare(track_dict):
    scores, colors, masks = [], [], []
    for id_, instance in track_dict.items():
        masks.append(mask_utils.decode(instance['segm']))
        colors.append([instance['label'] + 1, 0, id_ >> 8, id_ & 255])
        scores.append(instance['bbox'][-1])
    return scores, colors, masks


def mask_merge(mask_infor, img_name, bitmask_base):
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


def mask_merge_parallel(track_dicts, img_names, bitmask_base, nproc):
    with Pool(nproc) as pool:
        print('\nCollecting mask information')
        mask_infors = pool.map(mask_prepare, tqdm(track_dicts))
        print('\nMerging overlapped masks.')
        pool.starmap(
            partial(mask_merge, bitmask_base=bitmask_base),
            tqdm(zip(mask_infors, img_names), total=len(mask_infors)))
