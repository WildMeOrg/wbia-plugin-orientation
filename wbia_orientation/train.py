# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# Written by Olga Moskvyak (olga.moskvyak@hdr.qut.edu.au)
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
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
from utils.utils import create_logger
from dataset import custom_transforms as ctf

import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str
    )

    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    return args


def _make_model(cfg, is_train):
    """Initialise model from config
    Input:
        cfg: config object
    Returns:
        model: model object
    """
    model = models.orientation_net.OrientationNet(cfg, is_train)
    return model


def _model_to_gpu(model, cfg):
    if cfg.USE_GPU:
        model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    return model


def _make_loss(cfg):
    loss_func = nn.MSELoss()
    return loss_func


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
    train_tform = transforms.Compose(
        [
            ctf.RandomHorizontalFlip(p=cfg.DATASET.HOR_FLIP_PROB),
            ctf.RandomVerticalFlip(p=cfg.DATASET.VERT_FLIP_PROB),
            ctf.RandomRotate(degrees=cfg.DATASET.MAX_ROT, mode='edge'),
            ctf.RandomScale(scale=cfg.DATASET.SCALE_FACTOR),
            ctf.CropObjectAlignedArea(noise=0.1),
            ctf.Resize(cfg.MODEL.IMSIZE),
            ctf.ColorJitterSample(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
            ctf.ToTensor(),
            ctf.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                input_size=cfg.MODEL.IMSIZE[0],
            ),
        ]
    )

    valid_tform = transforms.Compose(
        [
            ctf.CropObjectAlignedArea(noise=0.0),
            ctf.Resize(cfg.MODEL.IMSIZE),
            ctf.ToTensor(),
            ctf.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                input_size=cfg.MODEL.IMSIZE[0],
            ),
        ]
    )

    train_dataset = eval('dataset.' + cfg.DATASET.CLASS)(cfg, True, train_tform)
    valid_dataset = eval('dataset.' + cfg.DATASET.CLASS)(cfg, False, valid_tform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BS * len(cfg.GPUS),
        shuffle=True,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BS * len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    return train_loader, valid_loader, valid_dataset


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, output_dir, tb_log_dir = create_logger(cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # Initialise models
    model = _make_model(cfg, is_train=True)

    # Initialise losses
    loss_func = _make_loss(cfg)

    # Initialise data loaders
    train_loader, valid_loader, valid_dataset = _make_data(cfg, logger)

    best_perf = 0.0
    is_best_model = False

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(output_dir, 'checkpoint.pth')

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        model.load_state_dict(checkpoint['state_dict'])
        logger.info(
            "=> loaded checkpoint '{}' (epoch {})".format(
                checkpoint_file, checkpoint['epoch']
            )
        )

    model = _model_to_gpu(model, cfg)
    optimizer = get_optimizer(cfg, model)

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        optimizer.load_state_dict(checkpoint['optimizer'])

    # Train epochs
    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):

        train(
            cfg, train_loader, model, loss_func, optimizer, epoch, output_dir, writer_dict
        )

        # Evaluate on validation set
        perf_indicator = validate(
            cfg, valid_loader, valid_dataset, model, loss_func, output_dir, writer_dict
        )

        if perf_indicator >= best_perf:
            best_perf = perf_indicator
            is_best_model = True
        else:
            is_best_model = False

        # Save checkpoint
        logger.info('=> saving checkpoint to {}'.format(output_dir))
        checkpoint_dict = {
            'epoch': epoch + 1,
            'model': cfg.MODEL.CORE_NAME,
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }
        checkpoint_dict['state_dict'] = (
            model.module.state_dict() if cfg.USE_GPU else model.state_dict()
        )

        # Save best model
        torch.save(checkpoint_dict, os.path.join(output_dir, 'checkpoint.pth'))
        if is_best_model:
            logger.info(
                '=> saving best model state to {} at epoch {}'.format(output_dir, epoch)
            )
            torch.save(
                checkpoint_dict['state_dict'], os.path.join(output_dir, 'best.pth')
            )

    # Save final state
    logger.info('=> saving final model state to {}'.format(output_dir))
    torch.save(checkpoint_dict['state_dict'], os.path.join(output_dir, 'final.pth'))

    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
