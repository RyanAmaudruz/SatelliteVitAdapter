# dataset settings
dataset_type = 'DFC2020Dataset'
# data_root = '/gpfs/work5/0/prjs0790/data/grss/dfc2020_mmseg'
data_root = '/var/node433/local/ryan_a/data/dfc2020_mmseg'
img_norm_cfg = dict(
    mean=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0 ,0 ,0], std=[10000,10000,10000,10000,10000,10000,10000,10000,10000,10000,10000,10000,10000], to_rgb=False)
crop_size = (256, 256)
train_pipeline = [
    dict(type='LoadImageFromFile_MS'),
    dict(type='LoadAnnotations',reduce_zero_label=True),
    dict(type='Resize', img_scale=(256, 256), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    #dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile_MS'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(256, 256),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='s2_dir/train',
        ann_dir='ann_dir/train',
        #split='train.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='s2_dir/test',
        ann_dir='ann_dir/test',
        #split='val.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='s2_dir/test',
        ann_dir='ann_dir/test',
        pipeline=test_pipeline))
