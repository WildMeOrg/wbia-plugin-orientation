# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# Written by Olga Moskvyak (olga.moskvyak@hdr.qut.edu.au)
# ------------------------------------------------------------------------------
import os
import time
import logging
import torch
import numpy as np

from core.evaluate import evaluate_orientaion_coords, compute_theta
from core.evaluate import evaluate_orientaion_theta
from utils.vis import plot_images, plot_images_theta, plot_rotated
from utils.utils import AverageMeterSet
from utils.utils import save_object

torch.autograd.set_detect_anomaly(True)
logger = logging.getLogger(__name__)


def train(cfg, train_loader, model, loss_func, optimizer, epoch, output_dir,
          writer_dict):

    meters = AverageMeterSet()

    # Switch to train mode
    model.train()

    # Iterate over data loader
    for i, (images, xc, yc, xt, yt, w, theta) in enumerate(train_loader):
        end = time.time()
        bs = images.size(0)
        if cfg.MODEL.PREDICT_THETA:
            target_output = torch.cos(theta).view(theta.size(0), 1)
        else:
            target_output = torch.stack([xc, yc, xt, yt, w], dim=1)

        if cfg.USE_GPU:
            images = images.cuda(non_blocking=True)
            target_output = target_output.cuda(non_blocking=True)

        # Compute output of Orientation Network
        output = model(images)

        # Compute loss and backpropagate
        loss = loss_func(output, target_output)
        meters.update('train_loss', loss.item(), bs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute performance: accuracy of theta and coords
        if cfg.MODEL.PREDICT_THETA:
            perf = evaluate_orientaion_theta(output.detach().cpu(),
                                             target_output.detach().cpu())
        else:
            perf = evaluate_orientaion_coords(output.detach().cpu(),
                                              target_output.detach().cpu(),
                                              theta,
                                              theta_thr=cfg.TEST.THETA_THR,
                                              theta_source='annot')
            meters.update('train_err_xcyc',  perf['err_xcyc'], bs)
            meters.update('train_err_xtyt',  perf['err_xtyt'], bs)
            meters.update('train_err_w', perf['err_w'], bs)

        meters.update('train_err_theta', perf['err_theta'], bs)
        meters.update('train_acc_theta', perf['acc_theta'], bs)

        # Measure elapsed time
        batch_time = time.time() - end

        if i % cfg.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\tBatch time {batch_time:.3f}s\t'. \
                            format(epoch, i, len(train_loader),
                                   batch_time=batch_time)
            for key, val in meters.meters.items():
                msg += '{} {:.4f} ({:.4f})\t'.format(key, val.val, val.avg)
            logger.info(msg)

            # Update tensorboard logs
            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']

            for key, val in meters.values().items():
                writer.add_scalar(key, val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            # Save some image with detected points
            if cfg.MODEL.PREDICT_THETA:
                plot_images_theta(images.cpu(),
                                  target_output.cpu(),
                                  output.detach().cpu(),
                                  '{}_{}'.format(cfg.DATASET.TRAIN_SET, i),
                                  output_dir)
            else:
                plot_images(images.cpu(),
                            target_output.cpu()*cfg.MODEL.IMSIZE[0],
                            output.detach().cpu()*cfg.MODEL.IMSIZE[0],
                            theta,
                            compute_theta(output.detach().cpu().numpy()),
                            '{}_{}'.format(cfg.DATASET.TRAIN_SET, i),
                            output_dir)

        if cfg.LOCAL and i > 3:
            break


def validate(cfg, val_loader, val_dataset, model, loss_func, output_dir,
             writer_dict=None):
    meters = AverageMeterSet()
    theta_gt_all = []
    theta_preds_all = []

    # Switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, xc, yc, xt, yt, w, theta) in enumerate(val_loader):
            bs = images.size(0)
            if cfg.MODEL.PREDICT_THETA:
                target_output = torch.cos(theta).view(theta.size(0), 1)
            else:
                target_output = torch.stack([xc, yc, xt, yt, w], dim=1)

            if cfg.USE_GPU:
                images = images.cuda(non_blocking=True)
                target_output = target_output.cuda(non_blocking=True)

            # Compute output of Orientation Network
            output = model(images)

            loss = loss_func(output, target_output)
            meters.update('valid_loss', loss.item(), bs)

            # Compute accuracy on all examples
            if cfg.MODEL.PREDICT_THETA:
                perf = evaluate_orientaion_theta(output.detach().cpu(),
                                                 target_output.detach().cpu())
            else:
                perf = evaluate_orientaion_coords(output.detach().cpu(),
                                                  target_output.detach().cpu(),
                                                  theta,
                                                  theta_thr=cfg.TEST.THETA_THR,
                                                  theta_source='annot')
                meters.update('valid_err_xcyc', perf['err_xcyc'], bs)
                meters.update('valid_err_xtyt', perf['err_xtyt'], bs)
                meters.update('valid_err_w', perf['err_w'], bs)

            meters.update('valid_err_theta', perf['err_theta'], bs)
            meters.update('valid_acc_theta', perf['acc_theta'], bs)

            theta_gt_all.append(theta.cpu().numpy())
            theta_preds_all.append(compute_theta(output.detach().cpu().numpy()))

            if i % cfg.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t'.format(i, len(val_loader))
                for key, val in meters.meters.items():
                    msg += '{} {:.4f} ({:.4f})\t'.format(key, val.val, val.avg)
                logger.info(msg)

            # Save some image with detected points
            if cfg.MODEL.PREDICT_THETA:
                plot_images_theta(images.cpu(),
                                  target_output.cpu(),
                                  output.detach().cpu(),
                                  '{}_{}'.format(cfg.DATASET.TEST_SET, i),
                                  output_dir)
            else:
                plot_images(images.cpu(),
                            target_output.cpu()*cfg.MODEL.IMSIZE[0],
                            output.detach().cpu()*cfg.MODEL.IMSIZE[0],
                            theta,
                            compute_theta(output.detach().cpu().numpy()),
                            '{}_{}'.format(cfg.DATASET.TEST_SET, i),
                            output_dir)
                # Plot rotated images
                if cfg.TEST.PLOT_ROTATED:
                    plot_rotated(images.cpu(),
                                 target_output.cpu()*cfg.MODEL.IMSIZE[0],
                                 output.detach().cpu()*cfg.MODEL.IMSIZE[0],
                                 theta,
                                 compute_theta(output.detach().cpu().numpy()),
                                 '{}_{}_rot'.format(cfg.DATASET.TEST_SET, i),
                                 output_dir)
            if cfg.LOCAL and i > 3:
                break

        # Update tensorboard
        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            for key, val in meters.values().items():
                writer.add_scalar(key, val, global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1

        # Save ground truth and predicted theta for analysis
        theta_gt_all = np.concatenate(theta_gt_all)
        theta_preds_all = np.concatenate(theta_preds_all)
        save_object(theta_gt_all, os.path.join(output_dir, 'theta_gt.pkl'))
        save_object(theta_preds_all, os.path.join(output_dir, 'theta_preds.pkl'))

    return meters.meters['valid_acc_theta'].avg
