# -*- coding: utf-8 -*-
'''
Author: Ioannis Kakogeorgiou
Email: gkakogeorgiou@gmail.com
Python Version: 3.7.10
Description: mados_evaluation.py includes the code in order to produce
             the evaluation for each class as well as the prediction
             masks for the pixel-level semantic segmentation.
'''

import os
import sys
import random
import logging
import rasterio
import torchvision
from mmcv.parallel import MMDataParallel, collate
from mmseg.apis import single_gpu_test
from mmseg.datasets import build_dataset, build_dataloader
from mmseg.models import build_segmentor
from rasterio.enums import Resampling
import argparse
from glob import glob
import numpy as np
import tifffile
from tqdm import tqdm
from os.path import dirname as up
sys.path.append('./')

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.nn import functional as F

from mmcv.utils import Config

# from segmentation.model_configuration import model_config
from segmentation.utils.assets import labels
from segmentation.utils.dataset import MADOS, bands_mean
import mmcv_custom   # noqa: F401,F403
import mmseg_custom   # noqa: F401,F403
from segmentation.utils.metrics import Evaluation, confusion_matrix
from segmentation.utils.test_time_aug import TTA




# sys.path.append(up(os.path.abspath(__file__)))

# # from marinext_wrapper import MariNext
#
# sys.path.append(os.path.join(up(up(os.path.abspath(__file__))), 'utils'))
# from dataset import MADOS, bands_mean, bands_std
# from test_time_aug import TTA
# from metrics import Evaluation, confusion_matrix
# from assets import labels, bool_flag

# root_path = up(up(os.path.abspath(__file__)))
#
# logging.basicConfig(filename=os.path.join(root_path, 'logs','evaluating_marinext.log'), filemode='a',level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
# logging.info('*'*10)

def seed_all(seed):
    # Pytorch Reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_band(path):
    return int(path.split('_')[-2])


def add_missing_chan(img):

    b, c, h, w = img.shape

    bands_mean = [0.0582676,  0.05223386, 0.04381474, 0.0357083,  0.03412902, 0.03680401,
                  0.03999107, 0.03566642, 0.03965081, 0.0267993,  0.01978944]

    temp_shape = list(img.shape)
    temp_shape[1] = 1
    impute_nan = torch.Tensor(bands_mean)[None, :, None, None].tile(temp_shape)

    cond = torch.isnan(img)

    img[cond] = impute_nan[cond]

    zeros = torch.zeros((b, 2, h, w), dtype=img.dtype)

    img = torch.cat([
        img[:, :9],
        zeros,
        img[:, 9:],
    ], 1)
    return img


def single_gpu_test_new(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False,
                    opacity=0.5,
                    pre_eval=False,
                    format_only=False,
                    format_args={}):
    """Test with single GPU by progressive mode.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during inference. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Mutually exclusive with
            pre_eval and format_results. Default: False.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
        pre_eval (bool): Use dataset.pre_eval() function to generate
            pre_results for metric evaluation. Mutually exclusive with
            efficient_test and format_results. Default: False.
        format_only (bool): Only format result for results commit.
            Mutually exclusive with pre_eval and efficient_test.
            Default: False.
        format_args (dict): The args for format_results. Default: {}.
    Returns:
        list: list of evaluation pre-results or list of save file names.
    """
    if efficient_test:
        warnings.warn(
            'DeprecationWarning: ``efficient_test`` will be deprecated, the '
            'evaluation is CPU memory friendly with pre_eval=True')
        mmcv.mkdir_or_exist('.efficient_test')
    # when none of them is set true, return segmentation results as
    # a list of np.array.
    assert [efficient_test, pre_eval, format_only].count(True) <= 1, \
        '``efficient_test``, ``pre_eval`` and ``format_only`` are mutually ' \
        'exclusive, only one of them could be true .'

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx
    loader_indices = data_loader.batch_sampler

    for batch_indices, data in zip(loader_indices, data_loader):

        with torch.no_grad():
            result = model(return_loss=False, **data)

        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result,
                    palette=dataset.PALETTE,
                    show=show,
                    out_file=out_file,
                    opacity=opacity)

        if efficient_test:
            result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]

        if format_only:
            result = dataset.format_results(
                result, indices=batch_indices, **format_args)
        if pre_eval:
            # TODO: adapt samples_per_gpu > 1.
            # only samples_per_gpu=1 valid now
            result = dataset.pre_eval(result, indices=batch_indices)
            results.extend(result)
        else:
            results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()

    return results


def main(options, cfg):
    # seed_all(0)
    # Transformations
    
    # transform_test = transforms.Compose([transforms.ToTensor()])
    # standardization = transforms.Normalize(bands_mean, bands_std)
    
    splits_path = os.path.join(options['path'],'splits')
    
    # Construct Data loader

    # dataset_test = MADOS(options['path'], splits_path, options['split'])
    #
    # test_loader = DataLoader(   dataset_test,
    #                             batch_size = options['batch'],
    #                             shuffle = False)

    val_dataset = build_dataset(cfg.data.test, dict(test_mode=True))

    val_dataloader = build_dataloader(
        val_dataset,
        samples_per_gpu=1,
        workers_per_gpu=8,
        dist=False,
        shuffle=False
    )

    # # Use gpu or cpu
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    # else:
    #     device = torch.device("cpu")
    #
    # models_list = []
    #
    # models_files = glob(os.path.join(options['model_path'],'*.pth'))
    #
    #
    # model_dir = '/gpfs/work5/0/prjs0790/data/mados_models/'
    # models_files = [model_dir + f for f in os.listdir(model_dir)]
    #
    # for model_file in models_files:
    
        # model = MariNext(options['input_channels'], options['output_channels'])

    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')
    )
    model.init_weights()
    
    model.to('cpu')

    # Load model from specific epoch to continue the training or start the evaluation

    # model_file = '/gpfs/work5/0/prjs0790/data/first_model_mados/upernet_deit_adapter_small_512_160k_mados/best_mIoU_iter_5000.pth'
    # model_file = '/var/node433/local/ryan_a/data/mados_fine_tuning/leo_new_trans_e09/best_mIoU_iter_33000.pth'
    # model_file = '/var/node433/local/ryan_a/data/mados_fine_tuning/new_queue-with_dino_loss_e04/best_mIoU_iter_34000.pth'
    model_file = '/var/node433/local/ryan_a/data/mados_fine_tuning/new_queue-with_dino_loss_e14/best_mIoU_iter_38000.pth'

    print(model_file)

    checkpoint = torch.load(model_file, map_location = 'cpu')

    # model.load_state_dict({f'module.{k}': v for k, v in checkpoint['state_dict'].items()})
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    model = MMDataParallel(
        model.cpu(), device_ids=[0])

    mode = 'test'


    y_true = []
    y_predicted = []

    true_neg_count = 0
    false_pos_count = 0

    for d in val_dataloader:
        # if options['test_time_augmentations'] and options['batch']==1: # Only with batch = 1
        #     image = TTA(image)
        #
        # image = add_missing_chan(image)
        #
        # image = image.to(device)
        # target = target.to(device)

        # seed_all(0)
        #
        # # all_predictions = []
        # image = torchvision.transforms.Resize(256)(image)

        # d['img'] = d['img'].to('cuda')

        with torch.no_grad():
            result = model(return_loss=False, **d)

        predictions = result[0]

        # if options['test_time_augmentations'] and options['batch']==1: # Only with batch = 1
        #     predictions = TTA(predictions, reverse_aggregation = True)

        crop_name = '_'.join(d['img_metas'][0].data[0][0]['ori_filename'].split('_')[:-1])

        target = tifffile.tifffile.imread(f'/var/node433/local/ryan_a/data/mados/mados_mmseg/ann_dir/{mode}/{crop_name}_ann.tif')
        target -= 1
        cond = target == -1
        target[cond] = 255

        predictions = predictions.reshape(-1)
        target = target.reshape(-1)
        mask = target != 255

        true_neg_bool = (target == 255) & (predictions == 255)
        true_neg_count += true_neg_bool.sum()

        false_pos_bool = (target == 255) & (predictions != 255)
        false_pos_count += false_pos_bool.sum()

        predictions = predictions[mask]
        target = target[mask]

        y_predicted += predictions.tolist()
        y_true += target.tolist()




    # return y_predicted, y_true

    ####################################################################
    # Save Scores to the .log file                                     #
    ####################################################################
    acc = Evaluation(y_predicted, y_true)
    print(f'True neg count: {true_neg_count}')
    print(f'False pos count: {false_pos_count}')
    print("\n")
    print("STATISTICS: \n")
    # print("Evaluation: " + str(acc))
    print("Evaluation: " + str(acc))
    conf_mat = confusion_matrix(y_true, y_predicted, labels, options['results_percentage'])
    # logging.info("Confusion Matrix:  \n" + str(conf_mat.to_string()))
    print("Confusion Matrix:  \n" + str(conf_mat.to_string()))



class FakeArgs:
    # path = '/gpfs/work5/0/prjs0790/data/mados/MADOS'
    # model_path = '/gpfs/work5/0/prjs0790/data/mados_models/1'
    path = '/var/node433/local/ryan_a/data/mados/MADOS'
    model_path = '/var/node433/local/ryan_a/data/mados_models/1'
    split = 'test'
    test_time_augmentations = True
    batch = 1
    input_channels = 11
    output_channels = 15
    # model_path = os.path.join(up(os.path.abspath(__file__)), 'trained_models', '45', 'model_ema.pth')
    predict_masks = False
    # gen_masks_path = os.path.join(root_path, 'data', 'predicted_marinext')
    results_percentage = True



if __name__ == "__main__":



    # parser = argparse.ArgumentParser()
    #
    # parser.add_argument('--path', help='Path of the images')
    # parser.add_argument('--split', default = 'test', type = str, help='Which dataset split (test or val)')
    # parser.add_argument('--test_time_augmentations', default= True, type=bool_flag, help='Generate maps and score based on multiple augmented testing samples? (Use batch = 1 !!!) ')
    #
    # parser.add_argument('--batch', default=1, type=int, help='Number of epochs to run')
	#
    # # Unet parameters
    # parser.add_argument('--input_channels', default=11, type=int, help='Number of input bands')
    # parser.add_argument('--output_channels', default=15, type=int, help='Number of output classes')
    #
    # # Unet model path
    # parser.add_argument('--model_path', default=os.path.join(up(os.path.abspath(__file__)), 'trained_models', '45', 'model_ema.pth'), help='Path to Unet pytorch model')
    # parser.add_argument('--results_percentage', default= True, type=bool_flag, help='Generate confusion matrix results in percentage?')
    #
    # # Produce Predicted Masks
    # parser.add_argument('--predict_masks', default= False, type=bool_flag, help='Generate test set prediction masks?')
    # parser.add_argument('--gen_masks_path', default=os.path.join(root_path, 'data', 'predicted_marinext'), help='Path to where to produce store predictions')

    args = FakeArgs()
    # options = vars(args)  # convert to ordinary dict
    options = {k: getattr(args, k) for k in dir(args) if not k.startswith('_')}

    config_path = '/var/node433/local/ryan_a/ViT-Adapter/segmentation/configs/s2c/upernet_deit_adapter_small_512_160k_mados.py'
    cfg = Config.fromfile(config_path)

    main(options, cfg)


    # print('t')
    #
    # acc = Evaluation(y_predicted, y_true)

    # collate
    #
    #
    # with torch.no_grad():
    #     t = model.forward(return_loss=False, img=d['img'], img_metas=d['img_metas'])
    #
    #
    #
    #
    #
    # results = single_gpu_test(
    #     model, val_dataloader, show=False, pre_eval=True
    # )


