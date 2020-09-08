# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random

import os, sys
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import kornia
from dataset.augmentation_func import RotationAug, PerspectiveAug

logger = logging.getLogger(__name__)


class KeypointDataset(Dataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        self.num_joints = 0
        #self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = []

        self.is_train = is_train
        self.root = root
        self.image_set = image_set

        self.output_path = cfg.OUTPUT_DIR
        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY
        self.prob_half_body = cfg.DATASET.PROB_HALF_BODY
        self.color_rgb = cfg.DATASET.COLOR_RGB

        self.target_type = cfg.MODEL.TARGET_TYPE
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        self.sigma = cfg.MODEL.SIGMA
        self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT
        self.joints_weight = 1

        self.transform = transform
        self.db = []
        
        if self.is_train:
            self.aug_func = self._init_aug_func(cfg)

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError
        
    def __len__(self,):
        return len(self.db)
    
    def _init_aug_func(self, cfg):
        if cfg.TRAIN.AUG_FUNC == 'rotation':
            return RotationAug(cfg.TRAIN.AUG_ROT, cfg.MODEL.IMAGE_SIZE, cfg.MODEL.HEATMAP_SIZE)
        elif cfg.TRAIN.AUG_FUNC == 'perspective':
            return PerspectiveAug(cfg.TRAIN.AUG_VAR, cfg.MODEL.IMAGE_SIZE, cfg.MODEL.HEATMAP_SIZE, cfg.DATASET.ROT_FACTOR)
        else:
            raise ValueError('Incorrect value of cfg.TRAIN_AUG_FUNC:', cfg.TRAIN.AUG_FUNC)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''

        if self.data_format == 'zip':
            from utils import zipreader
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
        else:
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))
            
        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']

        if self.transform:
            #Spatial transformations of image and corresponding landmark coordinates
            images, joints, joints_vis = self.transform((data_numpy, joints, joints_vis))          
        else:
            images = data_numpy
        
        #Transform coordinates to heatmaps (target) and visibility (target_weight)
        target, target_weight = self.generate_target(joints, joints_vis)
        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)
            
        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_vis': joints_vis
        }
        
        if self.is_train:
            #Augment image and record augmentation matrix for semi-supervised learning
            images_aug, target_aug, target_weight_aug, \
                inv_aug_matrix, inv_aug_matrix_hm = self.aug_func.augment(images.unsqueeze(0), 
                                                                          target.unsqueeze(0), 
                                                                          target_weight.unsqueeze(0))
                
            images_aug = images_aug.squeeze()
            target_aug = target_aug.squeeze()
            target_weight_aug = target_weight_aug.squeeze()
            target_weight_aug = target_weight_aug.unsqueeze(-1)
            inv_aug_matrix = inv_aug_matrix.squeeze()
            inv_aug_matrix_hm = inv_aug_matrix_hm.squeeze()
                        
            return images, target, target_weight, \
                    idx, meta, \
                    images_aug, target_aug, target_weight_aug, \
                    inv_aug_matrix, inv_aug_matrix_hm
        else:
            return images, target, target_weight, meta