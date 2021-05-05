_base_ = './qdtrack-hdf5.py'
# model settings
model = dict(
    type='EMQuasiDenseFasterRCNN',
    channels=256,
    proto_num=64,
    stage_num=3)

# dataset settings
dataset_type = 'EMBDDVideoDataset'
data_root = '/cluster/work/cvl/xiali/bdd100k/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(
        type='LoadMultiImagesFromFile',
        to_float32=True,
        file_client_args=dict(
            vid_db_path=data_root + 'hdf5s/track_val.hdf5',
            backend='hdf5')),
    dict(type='SeqResize', img_scale=(1296, 720), keep_ratio=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqPad', size_divisor=32),
    dict(type='SeqImageToTensor', keys=['img']),
    dict(type='SeqCollect', keys=['img'], ref_prefix='ref')
]
data = dict(
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'jsons/box_track_val_cocofmt.json',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'jsons/box_track_val_cocofmt.json',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(lr=0.02)
log_config = dict(interval=50)
