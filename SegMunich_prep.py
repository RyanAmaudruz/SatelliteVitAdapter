import shutil
from glob import glob
import numpy as np
import os

from tifffile import tifffile

#         target[target == 21] = 1
#         target[target == 22] = 2
#         target[target == 23] = 3
#         target[target == 31] = 4
#         target[target == 32] = 6
#         target[target == 33] = 7
#         target[target == 41] = 8
#         target[target == 13] = 9
#         target[target == 14] = 10

label_mapping = {
    21: 1,
    22: 2,
    23: 3,
    31: 4,
    32: 6,
    33: 7,
    41: 8,
    13: 9,
    14: 10
}



def prepare_dir(file_path):
    """
    This function is used to create the directories needed to output a path. If the directories already exist, the
    function continues.
    """
    # Remove the file name to only keep the directory path.
    dir_path = '/'.join(file_path.split('/')[:-1])
    # Try to create the directory. Will have no effect if the directory already exists.
    try:
        os.makedirs(dir_path)
    except FileExistsError:
        pass

def create_mmseg_dataset(split_path, data_path, out_path):

    for mode in ('train', 'val'):

        ROIs_split = np.genfromtxt(os.path.join(split_path, f'{mode}.txt'),dtype='str')

        t = set(ROIs_split)

        mode_path = f'{data_path}/{mode}'
        labels = glob(os.path.join(f'{mode_path}/label','*'))
        images = glob(os.path.join(f'{mode_path}/img','*'))
        code_to_label_file_map = {l.split('/')[-1].split('.')[0]: l for l in labels}
        code_to_img_file_map = {i.split('/')[-1].split('.')[0]: i for i in images}

        all_values = []

        i = 0

        for crop_name in ROIs_split:
            # img
            img_out_path = os.path.join(out_path, 'img_dir', mode, f'{crop_name}_img.tif')
            prepare_dir(img_out_path)

            shutil.copy(code_to_img_file_map[crop_name], img_out_path)
            # ann
            ann_out_path = os.path.join(out_path, 'ann_dir', mode, f'{crop_name}_ann.tif')
            prepare_dir(ann_out_path)

            im_raw = tifffile.imread(code_to_label_file_map[crop_name])

            for k, v in label_mapping.items():
                cond = im_raw == k
                im_raw[cond] = v
            im_final = im_raw + 1

            tifffile.imwrite(ann_out_path, im_final)

if __name__ == '__main__':
    create_mmseg_dataset(
        split_path='/var/node433/local/ryan_a/data/TUM_128/dataset',
        data_path='/var/node433/local/ryan_a/data/TUM_128',
        out_path='/var/node433/local/ryan_a/data/TUM_128/latest'
    )

#
#
# import os
# import tifffile
# import numpy as np
#
# data_dir_list = [
#     '/gpfs/work5/0/prjs0790/data/segmunich_new/segmunich_mmseg_test/ann_dir/train/',
#     '/gpfs/work5/0/prjs0790/data/segmunich_new/segmunich_mmseg_test/ann_dir/val/'
#     # '//var/node433/local/ryan_a/data/TUM_128/segmunich_mmseg/ann_dir/train/',
#     # '/var/node433/local/ryan_a/data/TUM_128/segmunich_mmseg/ann_dir/val/'
# ]
#
# for data_dir in data_dir_list:
#     all_uniques = []
#
#     for i, f in enumerate(os.listdir(data_dir)):
#         target = tifffile.tifffile.imread(data_dir + f)
#         target[target == 21] = 1
#         target[target == 22] = 2
#         target[target == 23] = 3
#         target[target == 31] = 4
#         target[target == 32] = 6
#         target[target == 33] = 7
#         target[target == 41] = 8
#         target[target == 13] = 9
#         target[target == 14] = 10
#
#         tifffile.tifffile.imwrite(data_dir + f, target)
#
#         if i % 100 == 0:
#             print(i)
#

# import os
# import tifffile
# import numpy as np
#
#
#
# data_dir_list = [
#     '/var/node433/local/ryan_a/data/TUM_128/segmunich_mmseg/ann_dir/train/',
#     '/var/node433/local/ryan_a/data/TUM_128/segmunich_mmseg/ann_dir/val/'
# ]
#
# all_labels = []
#
# for data_dir in data_dir_list:
#     all_uniques = []
#
#     for i, f in enumerate(os.listdir(data_dir)):
#         target = tifffile.tifffile.imread(data_dir + f)
#         target += 1
#         tifffile.tifffile.imwrite(data_dir + f.replace('_ann.tif', '_po_ann.tif'), target)
#         os.remove(data_dir + f)
#         if i % 100 == 0:
#             print(i)
