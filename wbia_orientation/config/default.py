# -*- coding: utf-8 -*-
# Written by Olga Moskvyak (olga.moskvyak@hdr.qut.edu.au)

from yacs.config import CfgNode as CN  # NOQA
from datetime import datetime
import os


MODULE_PATH = os.path.abspath(os.path.join(os.path.split(__file__)[0], '..'))

_C = CN()

_C.OUTPUT_DIR = os.path.join(MODULE_PATH, 'output')
_C.LOG_DIR = os.path.join(MODULE_PATH, 'log')
# _C.COCO_ANNOT_DIR = '/external/contractors/olga.moskvyak/data'
_C.COCO_ANNOT_DIR = os.path.join(MODULE_PATH, 'data')
_C.DATA_DIR = 'data'
_C.USE_GPU = True
_C.GPUS = (0,)
_C.WORKERS = 0
_C.PRINT_FREQ = 10
_C.AUTO_RESUME = True
_C.PIN_MEMORY = True
_C.RANK = 0
_C.VERSION = 'v0'
_C.LOCAL = False

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# Common params for models
_C.MODEL = CN()
_C.MODEL.CORE_NAME = 'hrnet'
_C.MODEL.PRETRAINED = os.path.join(
    MODULE_PATH, 'pretrained_models', 'hrnetv2_w32_imagenet_pretrained.pth'
)

_C.MODEL.IMSIZE = [224, 224]  # width * height, ex: 192 * 256
_C.MODEL.EXTRA = CN(new_allowed=True)

_C.MODEL.EXTRA.STAGE1 = CN()
_C.MODEL.EXTRA.STAGE1.NUM_MODULES = 1
_C.MODEL.EXTRA.STAGE1.NUM_RANCHES = 1
_C.MODEL.EXTRA.STAGE1.BLOCK = 'BOTTLENECK'
_C.MODEL.EXTRA.STAGE1.NUM_BLOCKS = [4]
_C.MODEL.EXTRA.STAGE1.NUM_CHANNELS = [64]
_C.MODEL.EXTRA.STAGE1.FUSE_METHOD = 'SUM'

_C.MODEL.EXTRA.STAGE2 = CN()
_C.MODEL.EXTRA.STAGE2.NUM_MODULES = 1
_C.MODEL.EXTRA.STAGE2.NUM_BRANCHES = 2
_C.MODEL.EXTRA.STAGE2.BLOCK = 'BASIC'
_C.MODEL.EXTRA.STAGE2.NUM_BLOCKS = [4, 4]
_C.MODEL.EXTRA.STAGE2.NUM_CHANNELS = [32, 64]
_C.MODEL.EXTRA.STAGE2.FUSE_METHOD = 'SUM'

_C.MODEL.EXTRA.STAGE3 = CN()
_C.MODEL.EXTRA.STAGE3.NUM_MODULES = 4
_C.MODEL.EXTRA.STAGE3.NUM_BRANCHES = 3
_C.MODEL.EXTRA.STAGE3.BLOCK = 'BASIC'
_C.MODEL.EXTRA.STAGE3.NUM_BLOCKS = [4, 4, 4]
_C.MODEL.EXTRA.STAGE3.NUM_CHANNELS = [32, 64, 128]
_C.MODEL.EXTRA.STAGE3.FUSE_METHOD = 'SUM'

_C.MODEL.EXTRA.STAGE4 = CN()
_C.MODEL.EXTRA.STAGE4.NUM_MODULES = 3
_C.MODEL.EXTRA.STAGE4.NUM_BRANCHES = 4
_C.MODEL.EXTRA.STAGE4.BLOCK = 'BASIC'
_C.MODEL.EXTRA.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
_C.MODEL.EXTRA.STAGE4.NUM_CHANNELS = [32, 64, 128, 256]
_C.MODEL.EXTRA.STAGE4.FUSE_METHOD = 'SUM'

_C.LOSS = CN()

# DATASET related params
_C.DATASET = CN()
_C.DATASET.NAME = ''
_C.DATASET.CLASS = 'animal'

today = datetime.today()
_C.DATASET.TRAIN_SET = 'train%d' % (today.year,)
_C.DATASET.VALID_SET = 'val%d' % (today.year,)
_C.DATASET.TEST_SET = 'test%d' % (today.year,)
_C.DATASET.DATA_FORMAT = 'jpg'
_C.DATASET.SELECT_CATS_LIST = []
_C.DATASET.SUFFIX = ''

# training data augmentation
_C.DATASET.HOR_FLIP_PROB = 0.5
_C.DATASET.VERT_FLIP_PROB = 0.5
_C.DATASET.SCALE_FACTOR = [0.8, 1.2]
_C.DATASET.MAX_ROT = 180

# train
_C.TRAIN = CN()
_C.TRAIN.BS = 48
_C.TRAIN.LR = 0.00005
_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [200]
_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0
_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 100
_C.TRAIN.RESUME = False
_C.TRAIN.CHECKPOINT = ''

# testing
_C.TEST = CN()
_C.TEST.BS = 48
_C.TEST.HFLIP = True
_C.TEST.VFLIP = True
_C.TEST.MODEL_FILE = ''
_C.TEST.THETA_THR = 10.0
_C.TEST.PLOT_ROTATED = True
_C.TEST.PLOT_ERRORS = True
_C.TEST.PLOT_ROTATED_PREDS_ONLY = True


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()


if __name__ == '__main__':
    import sys

    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
