import mmcv
from qdtrack.core.evaluation.mot import xyxy2xywh

pkl_file = './qdtrack-hdf5.pkl'
json_file = '/cluster/work/cvl/xiali/bdd100k/jsons/box_track_val_cocofmt.json'
bbox_file = '/cluster/work/cvl/xiali/bdd100k/jsons/qdtrack-bbox.json'
track_file = '/cluster/work/cvl/xiali/bdd100k/jsons/qdtrack-track.json'

res = mmcv.load(pkl_file)
coco = mmcv.load(json_file)


ann_id = 0
bbox_res = res['bbox_result']
coco['annotations'] = []

for img, bboxes_list in zip(coco['images'], bbox_res):
    img_id = img['id']
    for cls_, bboxes in enumerate(bboxes_list):
        for bbox in bboxes:
            ann_id += 1
            ann = {
                'image_id': img_id,
                'id': ann_id,
                'bbox': xyxy2xywh(bbox),
                'score': bbox[-1],
                'category_id': cls_ + 1,
            }
            coco['annotations'].append(ann)
mmcv.dump(coco, bbox_file)


ann_id = 0
coco['annotations'] = []
track_res = res['track_result']

for img, tracks in zip(coco['images'], track_res):
    img_id = img['id']
    for ins_id, obj in tracks.items():
        bbox = obj['bbox']
        ann = {
            'image_id': img_id,
            'id': ins_id,
            'bbox': xyxy2xywh(bbox),
            'score': bbox[-1],
            'category_id': obj['label'] + 1,
        }
        coco['annotations'].append(ann)
mmcv.dump(coco, track_file)
