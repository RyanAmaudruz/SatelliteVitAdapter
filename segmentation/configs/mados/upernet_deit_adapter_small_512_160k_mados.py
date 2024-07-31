# Copyright (c) Shanghai AI Lab. All rights reserved.
_base_ = [
    '../_base_/models/upernet_r50.py',
    # '../_base_/datasets/dfc2020.py',
    '../_base_/datasets/mados.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

# custom_imports = dict(
#     imports=['/gpfs/home2/ramaudruz/ViT-Adapter/segmentation/mmseg_custom/datasets/dfc2020.py'],
#     allow_failed_imports=False)
# pretrained = 'https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth'
pretrained = 'pretrained/deit_small_patch16_224-cd65a155.pth'
model = dict(
    pretrained=pretrained,
    backbone=dict(
        _delete_=True,
        type='ViTAdapter',
        patch_size=16,
        embed_dim=384,
        depth=12,
        in_chans=13,
        num_heads=6,
        mlp_ratio=4,
        drop_path_rate=0.2,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=6,
        cffn_ratio=0.25,
        deform_ratio=1.0,
        interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
        window_attn=[False] * 12,
        window_size=[None] * 12),
    decode_head=dict(num_classes=15, in_channels=[384, 384, 384, 384]),
    auxiliary_head=dict(num_classes=15, in_channels=384),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341))
)
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(2048, 512),
#         # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='ResizeToMultiple', size_divisor=32),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]
img_norm_cfg = dict(
    mean=[0.0582676,  0.05223386, 0.04381474, 0.0357083,  0.03412902, 0.03680401,
          0.03999107, 0.03566642, 0.03965081, 0.0267993,  0.01978944],
    std=[0.03240627, 0.03432253, 0.0354812,  0.0375769,  0.03785412, 0.04992323,
         0.05884482, 0.05545856, 0.06423746, 0.04211187, 0.03019115], to_rgb=False)
crop_size = (256, 256)
train_pipeline = [
    dict(type='LoadImageFromFile_MS_mados'),
    # dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
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
            # dict(type='RandomFlip'),
            # dict(type='Normalize', **img_norm_cfg),
            dict(type='AddMissingChannels_mados'),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
optimizer = dict(_delete_=True, type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01,
                 constructor='LayerDecayOptimizerConstructor',
                 paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.95, custom_keys={'head': dict(lr_mult=10.)})
                 )
lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=24,
          val=dict(pipeline=test_pipeline),
          test=dict(pipeline=test_pipeline))
runner = dict(type='IterBasedRunner')
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=1)
evaluation = dict(interval=1000, metric='mIoU', save_best='mIoU')
fp16 = dict(loss_scale=dict(init_scale=512))