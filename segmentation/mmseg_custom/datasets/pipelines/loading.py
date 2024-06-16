import os.path as osp

import mmcv
import numpy as np
from cv2 import imread
import tifffile
from mmseg.datasets import PIPELINES

import rasterio

# @PIPELINES.register_module(force=True)
# class LoadImageFromFile(object):
#     """Load an image from file.
#
#     Required keys are "img_prefix" and "img_info" (a dict that must contain the
#     key "filename"). Added or updated keys are "filename", "img", "img_shape",
#     "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
#     "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).
#
#     Args:
#         to_float32 (bool): Whether to convert the loaded image to a float32
#             numpy array. If set to False, the loaded image is an uint8 array.
#             Defaults to False.
#         color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
#             Defaults to 'color'.
#         file_client_args (dict): Arguments to instantiate a FileClient.
#             See :class:`mmcv.fileio.FileClient` for details.
#             Defaults to ``dict(backend='disk')``.
#         imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
#             'cv2'
#     """
#
#     def __init__(self,
#                  to_float32=False,
#                  color_type='color',
#                  file_client_args=dict(backend='disk'),
#                  imdecode_backend='cv2'):
#         self.to_float32 = to_float32
#         self.color_type = color_type
#         self.file_client_args = file_client_args.copy()
#         self.file_client = None
#         self.imdecode_backend = imdecode_backend
#
#     def __call__(self, results):
#         """Call functions to load image and get image meta information.
#
#         Args:
#             results (dict): Result dict from :obj:`mmseg.CustomDataset`.
#
#         Returns:
#             dict: The dict contains loaded image and meta information.
#         """
#
#         if self.file_client is None:
#             self.file_client = mmcv.FileClient(**self.file_client_args)
#
#         if results.get('img_prefix') is not None:
#             filename = osp.join(results['img_prefix'],
#                                 results['img_info']['filename'])
#         else:
#             filename = results['img_info']['filename']
#         img_bytes = self.file_client.get(filename)
#         img = mmcv.imfrombytes(
#             img_bytes, flag=self.color_type, backend=self.imdecode_backend)
#         if self.to_float32:
#             img = img.astype(np.float32)
#
#         results['filename'] = filename
#         results['ori_filename'] = results['img_info']['filename']
#         results['img'] = img
#         results['img_shape'] = img.shape
#         results['ori_shape'] = img.shape
#         # Set initial values for default meta_keys
#         results['pad_shape'] = img.shape
#         results['scale_factor'] = 1.0
#         num_channels = 1 if len(img.shape) < 3 else img.shape[2]
#         results['img_norm_cfg'] = dict(
#             mean=np.zeros(num_channels, dtype=np.float32),
#             std=np.ones(num_channels, dtype=np.float32),
#             to_rgb=False)
#         return results
#
#     def __repr__(self):
#         repr_str = self.__class__.__name__
#         repr_str += f'(to_float32={self.to_float32},'
#         repr_str += f"color_type='{self.color_type}',"
#         repr_str += f"imdecode_backend='{self.imdecode_backend}')"
#         return repr_str
#
#
# @PIPELINES.register_module(force=True)
# class LoadAnnotations(object):
#     """Load annotations for semantic segmentation.
#
#     Args:
#         reduce_zero_label (bool): Whether reduce all label value by 1.
#             Usually used for datasets where 0 is background label.
#             Default: False.
#         file_client_args (dict): Arguments to instantiate a FileClient.
#             See :class:`mmcv.fileio.FileClient` for details.
#             Defaults to ``dict(backend='disk')``.
#         imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
#             'pillow'
#     """
#
#     def __init__(self,
#                  reduce_zero_label=False,
#                  file_client_args=dict(backend='disk'),
#                  imdecode_backend='pillow'):
#         self.reduce_zero_label = reduce_zero_label
#         self.file_client_args = file_client_args.copy()
#         self.file_client = None
#         self.imdecode_backend = imdecode_backend
#
#     def __call__(self, results):
#         """Call function to load multiple types annotations.
#
#         Args:
#             results (dict): Result dict from :obj:`mmseg.CustomDataset`.
#
#         Returns:
#             dict: The dict contains loaded semantic segmentation annotations.
#         """
#
#         if self.file_client is None:
#             self.file_client = mmcv.FileClient(**self.file_client_args)
#
#         if results.get('seg_prefix', None) is not None:
#             filename = osp.join(results['seg_prefix'],
#                                 results['ann_info']['seg_map'])
#         else:
#             filename = results['ann_info']['seg_map']
#         img_bytes = self.file_client.get(filename)
#         gt_semantic_seg = mmcv.imfrombytes(
#             img_bytes, flag='unchanged',
#             backend=self.imdecode_backend).squeeze().astype(np.uint8)
#         # modify if custom classes
#         if results.get('label_map', None) is not None:
#             for old_id, new_id in results['label_map'].items():
#                 gt_semantic_seg[gt_semantic_seg == old_id] = new_id
#         # reduce zero_label
#         if self.reduce_zero_label:
#             # avoid using underflow conversion
#             gt_semantic_seg[gt_semantic_seg == 0] = 255
#             gt_semantic_seg = gt_semantic_seg - 1
#             gt_semantic_seg[gt_semantic_seg == 254] = 255
#         results['gt_semantic_seg'] = gt_semantic_seg
#         results['seg_fields'].append('gt_semantic_seg')
#         return results
#
#     def __repr__(self):
#         repr_str = self.__class__.__name__
#         repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
#         repr_str += f"imdecode_backend='{self.imdecode_backend}')"
#         return repr_str
#
#
# @PIPELINES.register_module(force=True)
# class LoadAnnotationsGTA(object):
#     """Load annotations for semantic segmentation.
#
#     Args:
#         reduce_zero_label (bool): Whether reduce all label value by 1.
#             Usually used for datasets where 0 is background label.
#             Default: False.
#         file_client_args (dict): Arguments to instantiate a FileClient.
#             See :class:`mmcv.fileio.FileClient` for details.
#             Defaults to ``dict(backend='disk')``.
#         imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
#             'pillow'
#     """
#
#     def __init__(self,
#                  reduce_zero_label=False,
#                  file_client_args=dict(backend='disk'),
#                  imdecode_backend='pillow'):
#         self.reduce_zero_label = reduce_zero_label
#         self.file_client_args = file_client_args.copy()
#         self.file_client = None
#         self.imdecode_backend = imdecode_backend
#
#     def __call__(self, results):
#         """Call function to load multiple types annotations.
#
#         Args:
#             results (dict): Result dict from :obj:`mmseg.CustomDataset`.
#
#         Returns:
#             dict: The dict contains loaded semantic segmentation annotations.
#         """
#         if self.file_client is None:
#             self.file_client = mmcv.FileClient(**self.file_client_args)
#
#         if results.get('seg_prefix', None) is not None:
#             filename = osp.join(results['seg_prefix'],
#                                 results['ann_info']['seg_map'])
#         else:
#             filename = results['ann_info']['seg_map']
#         #img_bytes = self.file_client.get(filename)
#         gt_semantic_seg = imread(filename, 2) / 100.
#         #gt_semantic_seg = imread(filename, 2)
#         gt_semantic_seg = np.clip(gt_semantic_seg, 0, 500)
#         if np.isnan(gt_semantic_seg.sum()):
#             gt_semantic_seg = np.where(np.isnan(gt_semantic_seg), np.full_like(gt_semantic_seg, 0), gt_semantic_seg)
#         # modify if custom classes
#         results['gt_semantic_seg'] = gt_semantic_seg
#         results['seg_fields'].append('gt_semantic_seg')
#         return results
#
#     def __repr__(self):
#         repr_str = self.__class__.__name__
#         repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
#         repr_str += f"imdecode_backend='{self.imdecode_backend}')"
#         return repr_str
#
#
#
# @PIPELINES.register_module(force=True)
# class LoadAnnotationsDepth(object):
#     """Load annotations for semantic segmentation.
#
#     Args:
#         reduce_zero_label (bool): Whether reduce all label value by 1.
#             Usually used for datasets where 0 is background label.
#             Default: False.
#         file_client_args (dict): Arguments to instantiate a FileClient.
#             See :class:`mmcv.fileio.FileClient` for details.
#             Defaults to ``dict(backend='disk')``.
#         imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
#             'pillow'
#     """
#
#     def __init__(self,
#                  reduce_zero_label=False,
#                  file_client_args=dict(backend='disk'),
#                  imdecode_backend='pillow'):
#         self.reduce_zero_label = reduce_zero_label
#         self.file_client_args = file_client_args.copy()
#         self.file_client = None
#         self.imdecode_backend = imdecode_backend
#
#     def __call__(self, results):
#         """Call function to load multiple types annotations.
#
#         Args:
#             results (dict): Result dict from :obj:`mmseg.CustomDataset`.
#
#         Returns:
#             dict: The dict contains loaded semantic segmentation annotations.
#         """
#
#         if self.file_client is None:
#             self.file_client = mmcv.FileClient(**self.file_client_args)
#
#         if results.get('seg_prefix', None) is not None:
#             filename = osp.join(results['seg_prefix'],
#                                 results['ann_info']['seg_map'])
#         else:
#             filename = results['ann_info']['seg_map']
#         #img_bytes = self.file_client.get(filename)
#
#         #filename = filename[:-7]+'.png'
#         filename = filename.replace('RGB','AGL')
#
#         gt_semantic_seg = imread(filename, 2)
#         #gt_semantic_seg = imread(filename, 2) / 100.
#         gt_semantic_seg[gt_semantic_seg>400] = 0
#         #gt_semantic_seg = mmcv.imread(filename,2)
#         gt_semantic_seg = np.clip(gt_semantic_seg, 0, 400)
#         # If these is NaN value
#         #if np.isnan(gt_semantic_seg.sum()):
#         #    gt_semantic_seg = np.where(np.isnan(gt_semantic_seg), np.full_like(gt_semantic_seg, 0), gt_semantic_seg)
#         '''gt_semantic_seg = mmcv.imfrombytes(
#             iimg_bytes, flag='unchanged',
#             backend=self.imdecode_backend).squeeze()'''
#         # modify if custom classes
#         results['gt_semantic_seg'] = gt_semantic_seg
#         results['seg_fields'].append('gt_semantic_seg')
#         return results
#
#     def __repr__(self):
#         repr_str = self.__class__.__name__
#         repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
#         repr_str += f"imdecode_backend='{self.imdecode_backend}')"
#         return repr_str

@PIPELINES.register_module(force=True)
class LoadImageFromFile_MS_mados(object):
    """Load a multispectral tif image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        #img_bytes = self.file_client.get(filename)
        #img = mmcv.imfrombytes(
        #    img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        if 'mados' in filename:
            img = tifffile.imread(filename)
            img = np.transpose(img,(1,2,0))
        else:
            with rasterio.open(filename,'r') as rf:
                img = rf.read() # (C,W,H)
                img = np.transpose(img,(1,2,0))
            if self.to_float32:
                img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str



@PIPELINES.register_module(force=True)
class LoadImageFromFile_MS_segmunich(object):
    """Load a multispectral tif image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        #img_bytes = self.file_client.get(filename)
        #img = mmcv.imfrombytes(
        #    img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        # if 'mados' in filename:
        img = tifffile.imread(filename)
        # img = np.transpose(img,(1,2,0))
        # else:
        #     with rasterio.open(filename,'r') as rf:
        #         img = rf.read() # (C,W,H)
        #         img = np.transpose(img,(1,2,0))
        #     if self.to_float32:
        #         img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str



# @PIPELINES.register_module(force=True)
# class LoadAnnotationsNew(object):
#     """Load annotations for semantic segmentation.
#
#     Args:
#         reduce_zero_label (bool): Whether reduce all label value by 1.
#             Usually used for datasets where 0 is background label.
#             Default: False.
#         file_client_args (dict): Arguments to instantiate a FileClient.
#             See :class:`mmcv.fileio.FileClient` for details.
#             Defaults to ``dict(backend='disk')``.
#         imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
#             'pillow'
#     """
#
#     def __init__(self,
#                  reduce_zero_label=False,
#                  file_client_args=dict(backend='disk'),
#                  imdecode_backend='pillow'):
#         self.reduce_zero_label = reduce_zero_label
#         self.file_client_args = file_client_args.copy()
#         self.file_client = None
#         self.imdecode_backend = imdecode_backend
#
#     def __call__(self, results):
#         """Call function to load multiple types annotations.
#
#         Args:
#             results (dict): Result dict from :obj:`mmseg.CustomDataset`.
#
#         Returns:
#             dict: The dict contains loaded semantic segmentation annotations.
#         """
#
#         if self.file_client is None:
#             self.file_client = mmcv.FileClient(**self.file_client_args)
#
#         if results.get('seg_prefix', None) is not None:
#             filename = osp.join(results['seg_prefix'],
#                                 results['ann_info']['seg_map'])
#         else:
#             filename = results['ann_info']['seg_map']
#         img_bytes = self.file_client.get(filename)
#         gt_semantic_seg = mmcv.imfrombytes(
#             img_bytes, flag='unchanged',
#             backend=self.imdecode_backend).squeeze().astype(np.uint8)
#         # modify if custom classes
#         if results.get('label_map', None) is not None:
#             for old_id, new_id in results['label_map'].items():
#                 gt_semantic_seg[gt_semantic_seg == old_id] = new_id
#         # reduce zero_label
#         if self.reduce_zero_label:
#             # avoid using underflow conversion
#             gt_semantic_seg[gt_semantic_seg == 0] = 255
#             gt_semantic_seg = gt_semantic_seg - 1
#             gt_semantic_seg[gt_semantic_seg == 254] = 255
#         results['gt_semantic_seg'] = gt_semantic_seg
#         results['seg_fields'].append('gt_semantic_seg')
#         return results
#
#     def __repr__(self):
#         repr_str = self.__class__.__name__
#         repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
#         repr_str += f"imdecode_backend='{self.imdecode_backend}')"
#         return repr_str

@PIPELINES.register_module(force=True)
class AddMissingChannels_mados(object):
    def __init__(self):
        pass

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """
        h, w = results['img'].shape[:2]

        bands_mean = [0.0582676,  0.05223386, 0.04381474, 0.0357083,  0.03412902, 0.03680401,
         0.03999107, 0.03566642, 0.03965081, 0.0267993,  0.01978944]

        impute_nan = np.tile(bands_mean, (h,w,1))

        cond = np.isnan(results['img'])

        results['img'][cond] = impute_nan[cond]

        zeros = np.zeros((h, w, 2), dtype=results['img'].dtype)

        results['img'] = np.concatenate([
            results['img'][:, :, :9],
            zeros,
            results['img'][:, :, 9:],
        ], 2)
        return results


@PIPELINES.register_module(force=True)
class AddMissingChannels_segmunich(object):
    def __init__(self):
        pass

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """
        h, w = results['img'].shape[:2]

        results['img'] = results['img'].astype('float32') / 10000

        # bands_mean = [0.0582676,  0.05223386, 0.04381474, 0.0357083,  0.03412902, 0.03680401,
        #               0.03999107, 0.03566642, 0.03965081, 0.0267993,  0.01978944]
        #
        # impute_nan = np.tile(bands_mean, (h,w,1))

        # cond = np.isnan(results['img'])
        #
        # results['img'][cond] = impute_nan[cond]

        zeros_1 = np.zeros((h, w, 1), dtype=results['img'].dtype)

        zeros_2 = np.zeros((h, w, 2), dtype=results['img'].dtype)

        results['img'] = np.concatenate([
            results['img'][:, :, :7],
            zeros_1,
            results['img'][:, :, 7:8],
            zeros_2,
            results['img'][:, :, 8:],
        ], 2)
        return results
