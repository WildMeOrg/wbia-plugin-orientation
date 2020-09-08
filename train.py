# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# Written by Olga Moskvyak (olga.moskvyak@hdr.qut.edu.au)
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
import shutil
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import tools._init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss, TripletLoss, KeypointSimilarityLoss
from core.function import train, validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary

import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    
    args = parser.parse_args()

    return args

def _make_model(cfg, exp_output_dir, is_train):
    """Initialise model from config
    Input:
        cfg: config object
    Returns:
        model: model object
    """
    pass

def _model_to_gpu(model, cfg):
    if cfg.USE_GPU:
       model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
        
    return models
    

def _make_loss(cfg, logger):
    """Define loss function (criterion)
    """
    pass

def _make_data(cfg, logger):
    """Initialise train and validation loaders as per config parameters
    Input:
        cfg: config object
        logger: logging object
    Returns:
        train_loader:
        valid_loader:
        valid_dataset:
    """
    train_transform = transforms.Compose([
                        dataset.transformers.Rotate(cfg.DATASET.ROT_FACTOR),
                        dataset.transformers.RandomHorizontalFlip(cfg.DATASET.FLIP_PROB, cfg.DATASET.SYMM_LDMARKS),
                        dataset.transformers.RandomCrop(cfg.MODEL.IMAGE_SIZE[0]),
                        dataset.transformers.ToTensor(),
                        dataset.transformers.Normalize(mean=[0.485, 0.456, 0.406], 
                                                       std =[0.229, 0.224, 0.225])
                        ])
                        
    valid_transform = transforms.Compose([
                        dataset.transformers.CenterCrop(cfg.MODEL.IMAGE_SIZE[0]),
                        dataset.transformers.ToTensor(),
                        dataset.transformers.Normalize(mean=[0.485, 0.456, 0.406], 
                                                       std =[0.229, 0.224, 0.225])
                        ])
                        
    train_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True, train_transform)
    
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False, valid_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                shuffle=True,
                                                num_workers=cfg.WORKERS,
                                                pin_memory=cfg.PIN_MEMORY
                                            )
    
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                batch_size=cfg.TEST.BS*len(cfg.GPUS),
                                                shuffle=False,
                                                num_workers=cfg.WORKERS,
                                                pin_memory=cfg.PIN_MEMORY
                                            )
    
    return train_loader, valid_loader, valid_dataset

def _make_optimizer(cfg, models_dict):
    
    optimizer = torch.optim.Adam(models_dict['spen'].parameters(), lr=cfg.TRAIN.LR)
    
    if cfg.LOSS.RECONSTRUCTION or cfg.LOSS.RECONSTRUCTION_CONS:
        optimizer = torch.optim.Adam(list(models_dict['spen'].parameters())
                                            + list(models_dict['srn'].parameters()), lr=cfg.TRAIN.LR)
    return optimizer


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # Initialise models
    model = _make_model(cfg, final_output_dir, True)
    
    # Initialise losses
    loss_dict = _make_loss(cfg, logger)    
    
    # Initialise data loaders
    train_loader, valid_loader, valid_dataset = _make_data(cfg, logger)

    best_perf = 0.0
    is_best_model = False
    last_epoch = -1
    
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )
    
    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))
        
    model = _model_to_gpu(model, cfg)
    optimizer = _make_optimizer(cfg, model)
    
    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        optimizer.load_state_dict(checkpoint['optimizer'])

   
    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):

        train(cfg, train_loader, 
              model, 
              loss_dict, 
              optimizer, epoch,
              final_output_dir, writer_dict)

        # evaluate on validation set
        perf_indicator = validate(
            cfg, valid_loader, valid_dataset, 
            model, 
            loss_dict,
            final_output_dir, writer_dict
        )

        if perf_indicator >= best_perf:
            best_perf = perf_indicator
            is_best_model = True
        else:
            is_best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        checkpoint_dict = {
            'epoch': epoch + 1,
            'model': cfg.MODEL.NAME,
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }
        checkpoint_dict['state_dict'] = model.module.state_dict() if cfg.USE_GPU else model.state_dict()
            
        torch.save(checkpoint_dict, os.path.join(final_output_dir, 'checkpoint.pth'))
        if is_best_model:
            logger.info('=> saving best model state to {} at epoch {}'.format(final_output_dir, epoch))
            torch.save(checkpoint_dict['state_dict'], os.path.join(final_output_dir, 'best.pth'))

    logger.info('=> saving final model state to {}'.format(final_output_dir))
    torch.save(checkpoint_dict['state_dict'], os.path.join(final_output_dir, 'final.pth'))
    
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
