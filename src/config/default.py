# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# Written by Olga Moskvyak (olga.moskvyak@hdr.qut.edu.au)
# ------------------------------------------------------------------------------
from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = 'output'
_C.LOG_DIR = 'log'
_C.COCO_ANNOT_DIR = 'data'
_C.DATA_DIR = 'data'
_C.USE_GPU = True
_C.GPUS = (0,)
_C.WORKERS = 4
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
_C.MODEL.CORE_NAME = 'resnet50'
_C.MODEL.PRETRAINED = ''
_C.MODEL.IMSIZE = [256, 256]  # width * height, ex: 192 * 256
_C.MODEL.PREDICT_THETA = False
_C.MODEL.EXTRA = CN(new_allowed=True)

_C.LOSS = CN()

# DATASET related params
_C.DATASET = CN()
_C.DATASET.NAME = ''
_C.DATASET.CLASS = 'animal'
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.TEST_SET = 'valid'
_C.DATASET.DATA_FORMAT = 'jpg'

# training data augmentation
_C.DATASET.HOR_FLIP_PROB = 0.
_C.DATASET.VERT_FLIP_PROB = 0.
_C.DATASET.SCALE_FACTOR = [1., 1.]
_C.DATASET.MAX_ROT = 30

# train
_C.TRAIN = CN()
_C.TRAIN.BS = 32
_C.TRAIN.LR = 0.001
_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0
_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 140
_C.TRAIN.RESUME = False
_C.TRAIN.CHECKPOINT = ''

# testing
_C.TEST = CN()
_C.TEST.BS = 32
_C.TEST.FLIP_TEST = False
_C.TEST.MODEL_FILE = ''
_C.TEST.THETA_THR = 10.
_C.TEST.PLOT_ROTATED = True

# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False
_C.DEBUG.SAVE_BATCH_IMAGES_GT = False
_C.DEBUG.SAVE_BATCH_IMAGES_PRED = False


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
