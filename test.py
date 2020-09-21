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
from core.function import train, validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary
from dataset import custom_transforms

import dataset
import models

from train import parse_args, _make_model, _model_to_gpu, _make_loss
  
def _make_test_data(cfg, logger):
    """Initialise train and validation loaders as per config parameters
    Input:
        cfg: config object
        logger: logging object
    Returns:
        test_loader: Data Loader over test dataset
        test_dataset: test dataset object
    """
                        
    test_transform = transforms.Compose([
                        custom_transforms.CropObjectAlignedArea(noise=0.),
                        custom_transforms.Resize(cfg.MODEL.IMAGE_SIZE),
                        custom_transforms.ToTensor(),
                        custom_transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                               std =[0.229, 0.224, 0.225],
                                               input_size=cfg.MODEL.IMAGE_SIZE[0])
                        ])
                        
    test_dataset = eval('dataset.'+cfg.DATASET.CLASS)(cfg, False, test_transform)
    
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=cfg.TEST.BS*len(cfg.GPUS),
                                                shuffle=False,
                                                num_workers=cfg.WORKERS,
                                                pin_memory=cfg.PIN_MEMORY
                                            )
    
    return test_loader, test_dataset


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # Initialise models
    model = _make_model(cfg)
    
    #Load model weights
    if cfg.TEST.MODEL_FILE:
        model_state_file = cfg.TEST.MODEL_FILE
    else:
        model_state_file = os.path.join(final_output_dir, 'best.pth')
        
    logger.info('=> loading model from {}'.format(model_state_file))     
    if cfg.USE_GPU:
        model.load_state_dict(torch.load(model_state_file))
    else:
        model.load_state_dict(torch.load(model_state_file, map_location=torch.device('cpu')))
    
    model = _model_to_gpu(model, cfg)
    # Initialise losses
    loss_func = _make_loss(cfg)
    
    # Initialise data loaders
    test_loader, test_dataset = _make_test_data(cfg, logger)

    # evaluate on validation set
    perf_indicator = validate(cfg, 
                              test_loader, 
                              test_dataset, 
                              model, 
                              loss_func,
                              final_output_dir)

    logger.info('Final results. Model performance on test data is {%.2f}'.format(perf_indicator))

if __name__ == '__main__':
    main()