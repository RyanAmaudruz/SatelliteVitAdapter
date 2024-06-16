from segmentation.mmseg_custom import DFC2020Dataset, madosDataset, segmunichDataset
from .wsdm2023_coco import WSDMCocoDataset
from .vg_dataset import VGDataset

__all__ = ['WSDMCocoDataset','VGDataset', 'DFC2020Dataset', 'madosDataset', 'segmunichDataset']
