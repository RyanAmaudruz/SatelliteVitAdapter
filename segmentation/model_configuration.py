
model_config = {
    'type': 'EncoderDecoder',
    'pretrained': None,
    'backbone': {
        'type': 'ViTAdapter',
        'patch_size': 16,
        'embed_dim': 384,
        'depth': 12,
        'in_chans': 13,
        'num_heads': 6,
        'mlp_ratio': 4,
        'drop_path_rate': 0.2,
        'conv_inplane': 64,
        'n_points': 4,
        'deform_num_heads': 6,
        'cffn_ratio': 0.25,
        'deform_ratio': 1.0,
        'interaction_indexes': [[0, 2], [3, 5], [6, 8], [9, 11]],
        'window_attn': [False, False, False, False, False, False, False, False, False, False, False, False],
        'window_size': [None, None, None, None, None, None, None, None, None, None, None, None]
    },
    'decode_head': {
        'type': 'UPerHead',
        'in_channels': [384, 384, 384, 384],
        'in_index': [0, 1, 2, 3],
        'pool_scales': (1, 2, 3, 6),
        'channels': 512, 'dropout_ratio': 0.1,
        'num_classes': 15,
        'norm_cfg': {'type': 'SyncBN', 'requires_grad': True},
        'align_corners': False,
        'loss_decode': {
            'type': 'CrossEntropyLoss', 'use_sigmoid': False, 'loss_weight': 1.0, 'class_weight': [
                43.306294917580665, 45.12108932219512, 43.306294917580665, 47.183574664875756, 36.53375506895221,
                5.373999539024519, 3.1931839168560345, 4.90035758550066, 48.99490439229447, 7.793912726634315,
                9.429311982230454, 30.717897644978887, 34.77940034479767, 46.454088480916994, 34.24714487198315
            ]
        }
    },
    'auxiliary_head': {
        'type': 'FCNHead',
        'in_channels': 384,
        'in_index': 2,
        'channels': 256,
        'num_convs': 1,
        'concat_input': False,
        'dropout_ratio': 0.1,
        'num_classes': 15,
        'norm_cfg': {'type': 'SyncBN', 'requires_grad': True},
        'align_corners': False,
        'loss_decode': {
            'type': 'CrossEntropyLoss', 'use_sigmoid': False, 'loss_weight': 0.4, 'class_weight': [
                43.306294917580665, 45.12108932219512, 43.306294917580665, 47.183574664875756, 36.53375506895221,
                5.373999539024519, 3.1931839168560345, 4.90035758550066, 48.99490439229447, 7.793912726634315,
                9.429311982230454, 30.717897644978887, 34.77940034479767, 46.454088480916994, 34.24714487198315
            ]
        }
    },
    'train_cfg': {},
    'test_cfg': {'mode': 'slide', 'crop_size': (512, 512), 'stride': (341, 341)}
}