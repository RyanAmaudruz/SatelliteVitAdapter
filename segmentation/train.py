# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings
import sys

import wandb


sys.path.append('segmentation/')

from utils.checkpoint_management import determine_weight_model, prepare_weights_for_vit_adapt


from api_train import init_random_seed, set_random_seed, train_segmentor

import mmcv
import mmcv_custom   # noqa: F401,F403
import mmseg_custom   # noqa: F401,F403
import torch
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import Config, DictAction, get_git_hash
from mmseg import __version__
# from mmseg.apis import init_random_seed, set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    # parser.add_argument('config', help='train config file path')
    # parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--load_from', help='the checkpoint file to load weights from', default=None
    )
    parser.add_argument(
        '--work_dir', help='the working directory to store files in', default=None
    )

    # parser.add_argument(
    #     '--resume-from', help='the checkpoint file to resume from')
    # parser.add_argument(
    #     '--no-validate',
    #     action='store_true',
    #     help='whether not to evaluate the checkpoint during training')
    # group_gpus = parser.add_mutually_exclusive_group()
    # group_gpus.add_argument(
    #     '--gpus',
    #     type=int,
    #     help='number of gpus to use '
    #     '(only applicable to non-distributed training)')
    # group_gpus.add_argument(
    #     '--gpu-ids',
    #     type=int,
    #     nargs='+',
    #     help='ids of gpus to use '
    #     '(only applicable to non-distributed training)')
    # parser.add_argument('--seed', type=int, default=None, help='random seed')
    # parser.add_argument(
    #     '--deterministic',
    #     action='store_true',
    #     help='whether to set deterministic options for CUDNN backend.')
    # parser.add_argument(
    #     '--options',
    #     nargs='+',
    #     action=DictAction,
    #     help="--options is deprecated in favor of --cfg_options' and it will "
    #     'not be supported in version v0.22.0. Override some settings in the '
    #     'used config, the key-value pair in xxx=yyy format will be merged '
    #     'into config file. If the value to be overwritten is a list, it '
    #     'should be like key="[a,b]" or key=a,b It also allows nested '
    #     'list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation '
    #     'marks are necessary and that no white space is allowed.')
    # parser.add_argument(
    #     '--cfg-options',
    #     nargs='+',
    #     action=DictAction,
    #     help='override some settings in the used config, the key-value pair '
    #     'in xxx=yyy format will be merged into config file. If the value to '
    #     'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
    #     'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
    #     'Note that the quotation marks are necessary and that no white space '
    #     'is allowed.')
    # parser.add_argument(
    #     '--launcher',
    #     choices=['none', 'pytorch', 'slurm', 'mpi'],
    #     default='none',
    #     help='job launcher')
    # parser.add_argument('--local_rank', type=int, default=0)
    # parser.add_argument(
    #     '--auto-resume',
    #     action='store_true',
    #     help='resume from the latest checkpoint automatically.')
    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)
    #
    # if args.options and args.cfg_options:
    #     raise ValueError(
    #         '--options and --cfg-options cannot be both '
    #         'specified, --options is deprecated in favor of --cfg-options. '
    #         '--options will not be supported in version v0.22.0.')
    # if args.options:
    #     warnings.warn('--options is deprecated in favor of --cfg-options. '
    #                   '--options will not be supported in version v0.22.0.')
    #     args.cfg_options = args.options

    return args

class FakeArgs:
    # config = '/var/node433/local/ryan_a/ViT-Adapter/segmentation/configs/s2c/upernet_deit_adapter_small_512_160k_mados.py'
    config = '/var/node433/local/ryan_a/ViT-Adapter/segmentation/configs/s2c/upernet_deit_adapter_small_512_160k_s2c.py'

    # config = None
    cfg_options = None
    work_dir = "/var/node433/local/ryan_a/data/mados_fine_tuning/transform_fixed-mixed_aug_e04_t"
    # work_dir = '/gpfs/work5/0/prjs0790/data/run_outputs/checkpoints/test'
    # work_dir = None
    load_from = None
    resume_from = None
    gpu = 0
    gpus = 1
    gpu_ids = None
    auto_resume = True
    launcher = 'none'
    seed = 0
    diff_seed = None
    deterministic = True
    validate = True
    no_validate = not validate

def main():

    pre_args = parse_args()

    args = FakeArgs()

    # pretrained_weights = '/var/node433/local/ryan_a/data/old_checkpoints/B13_vits16_dino_0099_ckpt.pth'
    # pretrained_weights = '/var/node433/local/ryan_a/data/ssl4eo_ssl/ssl4eo_ssl/ssl_s2c_new_transforms/checkpoint.pth'
    # pretrained_weights = '/var/node433/local/ryan_a/data/leo_missing/new_queue-with_dino_loss/20240419-002419_ckp-epoch=24_mod.ckpt'
    # pretrained_weights = '/var/node433/local/ryan_a/data/leo_missing/leopart_new_transform_leopart-20240221-081849/ckp-epoch=09_mod.ckpt'
    # pretrained_weights = '/var/node433/local/ryan_a/data/odin_missing_runs/transform_fixed-mixed_aug-w_local_negs/2024-04-06_12-31_ckp-epoch=04.ckpt'
    # pretrained_weights = '/gpfs/work5/0/prjs0790/data/run_outputs/checkpoints/ssl4eo_ssl/distillation_l2_normalised/checkpoint.pth'
    # pretrained_weights = '/var/node433/local/ryan_a/data/leo_missing/leo_new_queue/ckp-epoch=24_mod.ckpt'
    pretrained_weights = '/var/node433/local/ryan_a/data/odin_missing_runs/transform_fixed-mixed_aug/2024-04-06_08-14_ckp-epoch=04.ckpt'


    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg['runner']['max_iters'] = 40_000

    run = wandb.init(
        # Set the project where this run will be logged
        project="ViT-Adapter",
        name='train_vit_adapter',
        # Track hyperparameters and run metadata
        config={k: getattr(args, k) for k in dir(args) if not k.startswith('_')},
    )

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
    cfg.auto_resume = args.auto_resume

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # gpu_ids is used to calculate iter when resuming checkpoint
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    cfg.device = 'cuda'  # fix 'ConfigDict' object has no attribute 'device'
    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    # cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    # logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    print('Environment info:\n' + dash_line + env_info + '\n' +dash_line)
    meta['env_info'] = env_info

    # log some basic info
    # logger.info(f'Distributed training: {distributed}')
    # logger.info(f'Config:\n{cfg}')
    print(f'Distributed training: {distributed}')
    print(f'Config:\n{cfg}')

    # set random seeds
    seed = init_random_seed(args.seed)
    # logger.info(f'Set random seed to {seed}, 'f'deterministic: {args.deterministic}')
    print(f'Set random seed to {seed}, 'f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    if pre_args.load_from is not None:
        state_dict = torch.load(pre_args.load_from, map_location="cpu")
    else:
        state_dict = torch.load(pretrained_weights, map_location="cpu")

    weight_model = determine_weight_model(state_dict)
    state_dict = prepare_weights_for_vit_adapt(state_dict, weight_model)

    msg = model.load_state_dict(state_dict, strict=False)

    print(f'Pretrained weights: {pretrained_weights}')
    print(f'Missing keys: {msg.missing_keys}')
    print(f'Unexpected keys: {msg.unexpected_keys}')
    # print(f'Pretrained weights count: {len(pretrained_weights)}')
    print(f'Missing keys count: {len(msg.missing_keys)}')
    print(f'Unexpected keys count: {len(msg.unexpected_keys)}')

    if len(msg.missing_keys) > 322:
        raise ValueError('Too many missing key. Check the pretrained weight keys! Aborting!')

    # SyncBN is not support for DP
    if not distributed:
        warnings.warn(
            'SyncBN is only supported with DDP. To be compatible with DP, '
            'we convert SyncBN to BN. Please use dist_train.sh which can '
            'avoid this error.')
        model = revert_sync_batchnorm(model)

    # logger.info(model)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmseg version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            # config=cfg.pretty_text,
            config=str(cfg),
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    # passing checkpoint meta for saving best checkpoint
    meta.update(cfg.checkpoint_config.meta)
    cfg.log_config.hooks = [
        dict(type='TextLoggerHook'),
        dict(type='WandbHookSeg',
             init_kwargs={'project': 'ViT-Adapter'})
    ]

    # params_w_existing_weights = set([n for n, _ in model.named_parameters()]) - set(msg.missing_keys)
    # for n, p in model.named_parameters():
    #     if n in params_w_existing_weights:
    #         print(f'Setting no gradients for {n}')
    #         p.requires_grad =  False

    pretrained_keys = None

    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta,
        pretrained_keys=pretrained_keys,
    )


if __name__ == '__main__':
    main()
