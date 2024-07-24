# model settings
import numpy as np
# MADOS class distribution
class_dist = [
    0.00336, 0.00241, 0.00336, 0.00142, 0.00775, 0.18452, 0.34775, 0.20638, 0.00062, 0.1169, 0.09188, 0.01309, 0.00917,
    0.00176, 0.00963
]
# Segmunich class distribution
# class_dist = [
#     0.1819087, 0.244692, 0.0154577, 0.1137857, 0.28226, 0.017560357, 0.02637844, 0.008453476, 0.009553523, 0.0053915823, 0.01950901, 0.062555, 0.0124880
# ]
# # DFC2020 class distribution
# class_dist = [
#     0.209744, 0.0067099, 0.1465833, 0.016390, 0.2295087, 0.1316990, 0.00074441, 0.2586201
# ]
class_weight = [
    1/np.log(1.02 + d) for d in class_dist
    # (1/np.log(1.02 + d)) ** 1.2 for d in class_dist
]
# class_weight = [
#     # 1/np.log(1.02 + d) for d in class_dist
#     1
# ] * 13
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='UPerHead',
        in_channels=[256, 512, 1024, 2048],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=13,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,
            class_weight=class_weight
        )
        ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=15,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4,
            class_weight=class_weight
        )
        ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
