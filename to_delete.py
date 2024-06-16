import os

import pandas as pd
import tifffile
import numpy as np



# data_dir = '/gpfs/work5/0/prjs0790/data/segmunich_new/TUM_128/train/img/'
# f_list = os.listdir(data_dir)
#
#
# target = tifffile.tifffile.imread(f'{data_dir}{f_list[0]}')

# data_dir = '/gpfs/work5/0/prjs0790/data/segmunich_new/TUM_128/latest/ann_dir/val/'
#
# all_uniques = []
#
# for i, f in enumerate(os.listdir(data_dir)):
#
#     target = tifffile.tifffile.imread(data_dir + f)
#
#     all_uniques += list(np.unique(target))
#
#     if i > 1000:
#         break
#
# all_uniques_u = list(set(all_uniques))
#
# target = tifffile.tifffile.imread(f'{data_dir}{f_list[0]}')
#
# data_dir = '/gpfs/work5/0/prjs0790/data/segmunich_new/segmunich_mmseg_test/ann_dir/val/'
#
# data_dir_list = [
#     '/gpfs/work5/0/prjs0790/data/segmunich_new/segmunich_mmseg_test/ann_dir/train/',
#     '/gpfs/work5/0/prjs0790/data/segmunich_new/segmunich_mmseg_test/ann_dir/val/'
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
#         tifffile.tifffile.imwrite(data_dir + f, target)
#
#         if i % 100 == 0:
#             print(i)


# data_dir_list = [
#     '/gpfs/work5/0/prjs0790/data/segmunich_new/segmunich_mmseg_test/ann_dir/train/',
#     '/gpfs/work5/0/prjs0790/data/segmunich_new/segmunich_mmseg_test/ann_dir/val/'
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
#         tifffile.tifffile.imwrite(data_dir + f.replace('_po_ann.tif', '_ann.tif'), target)
#         os.remove(data_dir + f)
#         if i % 100 == 0:
#             print(i)


# import pandas as pd
#
# t10 = pd.Series(all_labels).value_counts()
#
#
# print(t10 / t10.sum())


all_targets = {}
data_dir = '/gpfs/work5/0/prjs0790/data/segmunich_new/TUM_128/latest/ann_dir/train/'
for i, f in enumerate(os.listdir(data_dir)):
    target = tifffile.imread(data_dir + f)
    # all_targets += list(target.flatten())
    # break
    value_dic = pd.Series((target.flatten())).value_counts()
    for k, v in value_dic.items():
        if k in all_targets:
            all_targets[k] += v
        else:
            all_targets[k] = v

    if i % 100 == 0:
        print(i)
        temp_sum = sum(v for v in all_targets.values())
        all_targets_ratio = {k: v/temp_sum for k, v in all_targets.items()}
        print(all_targets_ratio)


print(all_targets)

stat_tracker = None

all_targets = {}
data_dir = '/gpfs/work5/0/prjs0790/data/segmunich_new/TUM_128/latest/img_dir/train/'
for i, f in enumerate(os.listdir(data_dir)):
    target = tifffile.imread(data_dir + f)

    if i == 0:
        stat_tracker = target.mean((0, 1))
    else:
        stat_tracker = (i * stat_tracker + target.mean((0, 1))) / (i + 1)
    print(stat_tracker)

    i





