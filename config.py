import os

from yacs.config import CfgNode as CN

cfg = CN(new_allowed=True)
# ---------------------------------------------------------------------------- #
# TASK
# 0->cls, 1->seg
# ---------------------------------------------------------------------------- #
cfg.TASK = CN(new_allowed=True)
cfg.TASK.STATUS = 'train'
cfg.TASK.TYPE = 0  # 0 for classification, 1 for segmentation
cfg.TASK.NAME = 'aneurysm_cls'

cfg.SEED = 1234
cfg.METRICS = ['Acc', 'PR', 'NR']

cfg.MODEL = CN(new_allowed=True)
cfg.MODEL.NAME = 'resnet'
cfg.MODEL.DIM = '3d'
cfg.MODEL.BN = 'bn'
cfg.MODEL.INPUT_CHANNEL = 1
cfg.MODEL.NCLASS = 2
cfg.MODEL.PRETRAIN = ''
cfg.MODEL.DEEP_SUPERVISION = False

cfg.MODEL.BACKBONE = CN(new_allowed=True)
cfg.MODEL.BACKBONE.ARCH = 'resnet34'
cfg.MODEL.BACKBONE.HEAD = 'A'
# cfg.MODEL.BACKBONE.PRETRAIN = ''

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
cfg.SOLVER = CN(new_allowed=True)
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.LR_MODE = 'poly'
cfg.SOLVER.EPOCHS = 120
cfg.SOLVER.OPTIMIZER = CN(new_allowed=True)


# ---------------------------------------------------------------------------- #
# Loss
# ---------------------------------------------------------------------------- #
cfg.LOSS = CN(new_allowed=True)
cfg.LOSS.TYPE = 'ce_loss'
cfg.LOSS.CLASS_WEIGHT = []
cfg.LOSS.WEIGHT = []
cfg.LOSS.IGNORE_INDEX = -100
cfg.LOSS.DICE_WEIGHT = []

# ---------------------------------------------------------------------------- #
# Train
# ---------------------------------------------------------------------------- #
cfg.TRAIN = CN(new_allowed=True)
cfg.TRAIN.RESUME = False
cfg.TRAIN.PRINT = 50
cfg.TRAIN.DATA = CN(new_allowed=True)
cfg.TRAIN.DATA.WORKERS = 16
cfg.TRAIN.DATA.TRAIN_LIST = '/data3/pancw/data/patch/dataset/train/train.lst'
cfg.TRAIN.DATA.VAL_LIST = '/data3/pancw/data/patch/dataset/train/test.lst'
cfg.TRAIN.DATA.BATCH_SIZE = 32

# ---------------------------------------------------------------------------- #
# Test
# ---------------------------------------------------------------------------- #
cfg.TEST = CN(new_allowed=True)
cfg.TEST.MODEL_PTH = ' '
cfg.TEST.SAVE = True
cfg.TEST.SAVE_DIR = ' '
cfg.TEST.DATA = CN(new_allowed=True)
cfg.TEST.DATA.TEST_FILE = ' '
cfg.TEST.DATA.TEST_LIST = []
cfg.TEST.DATA.WORKERS = 16
cfg.TEST.DATA.BATCH_SIZE = 32
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
cfg.OUTPUT_DIR = "resnet34_ce_loss"
cfg.SAVE_ALL = False
# cfg.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")
