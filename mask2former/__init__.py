# Copyright (c) Facebook, Inc. and its affiliates.
from . import data  # register all new datasets
from . import modeling
# config
from .config import add_maskformer2_config
# dataset loading
from .data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import COCOInstanceNewBaselineDatasetMapper
from .data.dataset_mappers.mask_former_instance_dataset_mapper import (
    MaskFormerInstanceDatasetMapper,
)

from .evaluation.ins_seg_evaluation_zsi import InsSegzsiEvaluator
from .d2zero_model import D2Zero
