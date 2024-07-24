# dataset settings
dataset_type = 'madosDataset'
data_root = '/var/node433/local/ryan_a/data/mados/mados_mmseg'
# img_norm_cfg = dict(
#     mean=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0 ,0 ,0], std=[10000,10000,10000,10000,10000,10000,10000,10000,10000,10000,10000,10000,10000], to_rgb=False)

img_norm_cfg = dict(
    mean=[0.0582676,  0.05223386, 0.04381474, 0.0357083,  0.03412902, 0.03680401,
          0.03999107, 0.03566642, 0.03965081, 0.0267993,  0.01978944],
    std=[0.03240627, 0.03432253, 0.0354812,  0.0375769,  0.03785412, 0.04992323,
         0.05884482, 0.05545856, 0.06423746, 0.04211187, 0.03019115], to_rgb=False)
crop_size = (256, 256)
train_pipeline = [
    dict(type='LoadImageFromFile_MS_mados'),
    dict(type='LoadAnnotations',reduce_zero_label=True),
    dict(type='Resize', img_scale=(256, 256), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    #dict(type='PhotoMetricDistortion'),
    # dict(type='Normalize', **img_norm_cfg),
    dict(type='AddMissingChannels_mados'),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile_MS_mados'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(256, 256),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            # dict(type='Normalize', **img_norm_cfg),
            dict(type='AddMissingChannels_mados'),
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
        img_dir='img_dir/train',
        ann_dir='ann_dir/train',
        #split='train.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        #split='val.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/test',
        ann_dir='ann_dir/test',
        pipeline=test_pipeline))
