# -*- coding: utf-8 -*-
# Written by Olga Moskvyak (olga.moskvyak@hdr.qut.edu.au)

import math
import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from skimage import transform
from utils.utils import unnormalize
from utils.data_manipulation import plot_image_coordinates
from core.evaluate import normalize_theta
import matplotlib.style as style


def plot_boxes_gt_preds(
    input_images,
    coords_gt,
    coords_pred,
    theta_gt,
    theta_pred,
    prefix,
    output_dir,
    min_rows=2,
    max_cols=4,
    max_rows=4,
):
    """
    Plot grouhd truth and predicted object-aligned bounding boxes
    Args:
        input_images (torch tensor): shape (bs, c, h, w)
                                     images
        coords_gt (torch tensor): shape (bs, 5)
                   each row is [xc, yc, xt, yt, w] is ground truth
        coords_pred (torch tensor): shape (bs, 5)
                     each row is [xc, yc, xt, yt, w] is prediction
        theta_gt (torch tensor): shape (bs) with ground truth values of theta in radians
        theta_pred (torch tensor): shape (bs) with predictions of theta in radians
        prefix (string): prefix for plot filename
        output_dir (string): path to output directory to save plot
        min_rows (int): min number of rows in plot (default=2)
        max_cols (int): max number of columns in plot (default=4)
        max_rows (int): max number of rows in plot (default=4)
    """
    images_un = unnormalize(input_images)
    bs = input_images.size(0)
    ncols = min(max_cols, bs)
    nrows = int(math.ceil(float(bs) / ncols))
    if nrows == 1:
        nrows = 2
        ncols = bs // nrows
    nrows = min(nrows, max_rows)

    fig, ax = plt.subplots(
        nrows=nrows, ncols=ncols * 2, figsize=(ncols * 2 * 4, nrows * 4)
    )
    for r in range(nrows):
        for c in range(ncols):
            # Plot grouhd truth
            if r * ncols + c >= bs:
                continue
            plot_image_coordinates(
                ax[r, 2 * c],
                images_un[r * ncols + c].numpy().transpose((1, 2, 0)),
                coords_gt[r * ncols + c, 0].numpy(),
                coords_gt[r * ncols + c, 1].numpy(),
                coords_gt[r * ncols + c, 2].numpy(),
                coords_gt[r * ncols + c, 3].numpy(),
                coords_gt[r * ncols + c, 4].numpy(),
            )
            ax[r, 2 * c].set_title(
                'GT Theta {:.0f} deg'.format(math.degrees(theta_gt[r * ncols + c]))
            )
            ax[r, 2 * c].axis('off')

            # Plot predictions
            plot_image_coordinates(
                ax[r, 2 * c + 1],
                images_un[r * ncols + c].numpy().transpose((1, 2, 0)),
                coords_pred[r * ncols + c, 0].numpy(),
                coords_pred[r * ncols + c, 1].numpy(),
                coords_pred[r * ncols + c, 2].numpy(),
                coords_pred[r * ncols + c, 3].numpy(),
                coords_pred[r * ncols + c, 4].numpy(),
            )
            ax[r, 2 * c + 1].set_title(
                'Preds Theta {:.0f} deg'.format(math.degrees(theta_pred[r * ncols + c]))
            )
            ax[r, 2 * c + 1].axis('off')

    # Save plot
    file_name = os.path.join(output_dir, 'debug_images', '{}.png'.format(prefix))
    fig.savefig(file_name, format='png', dpi=100, bbox_inches='tight', facecolor='w')
    plt.close(fig)


def plot_rotated_gt_preds(
    input_images,
    coords_gt,
    coords_pred,
    theta_gt,
    theta_pred,
    prefix,
    output_dir,
    min_rows=2,
    max_cols=4,
    max_rows=4,
):
    """
    Plot images rotated with ground truth and predicted angles
    Args:
        input_images (torch tensor): shape (bs, c, h, w)
                                     images
        coords_gt (torch tensor): shape (bs, 5)
                   each row is [xc, yc, xt, yt, w] is ground truth
        coords_pred (torch tensor): shape (bs, 5)
                     each row is [xc, yc, xt, yt, w] is prediction
        theta_gt (torch tensor): shape (bs) with ground truth values of theta in radians
        theta_pred (torch tensor): shape (bs) with predictions of theta in radians
        prefix (string): prefix for plot filename
        output_dir (string): path to output directory to save plot
        min_rows (int): min number of rows in plot (default=2)
        max_cols (int): max number of columns in plot (default=4)
        max_rows (int): max number of rows in plot (default=4)
    """
    images_un = unnormalize(input_images).numpy().transpose(0, 2, 3, 1)
    coords_gt = coords_gt.numpy()
    coords_pred = coords_pred.numpy()
    bs = input_images.size(0)
    ncols = min(max_cols, bs)
    nrows = int(math.ceil(float(bs) / ncols))
    if nrows == 1:
        nrows = 2
        ncols = bs // nrows
    nrows = min(nrows, max_rows)

    fig, ax = plt.subplots(
        nrows=nrows, ncols=ncols * 2, figsize=(ncols * 2 * 4, nrows * 4)
    )
    for r in range(nrows):
        for c in range(ncols):
            # Plot images rotated by grouhd truth
            if r * ncols + c >= bs:
                continue
            degrees_gt = math.degrees(theta_gt[r * ncols + c])
            image_rotated = transform.rotate(
                images_un[r * ncols + c],
                angle=degrees_gt,
                center=coords_gt[r * ncols + c, :1],
            )
            ax[r, 2 * c].imshow(image_rotated)
            ax[r, 2 * c].set_title('GT Rotated by {:.0f} deg'.format(degrees_gt))
            ax[r, 2 * c].axis('off')

            # Plot predictions
            degrees_pred = math.degrees(theta_pred[r * ncols + c])
            image_rotated = transform.rotate(
                images_un[r * ncols + c],
                angle=degrees_pred,
                center=coords_pred[r * ncols + c, :1],
            )
            ax[r, 2 * c + 1].imshow(image_rotated)
            ax[r, 2 * c + 1].set_title('Preds Rotated by {:.0f} deg'.format(degrees_pred))
            ax[r, 2 * c + 1].axis('off')

    # Save plot
    file_name = os.path.join(output_dir, 'debug_images', '{}.png'.format(prefix))
    fig.savefig(file_name, format='png', dpi=100, bbox_inches='tight', facecolor='w')
    plt.close(fig)


def plot_rotated_preds(
    input_images,
    coords_pred,
    theta_pred,
    prefix,
    output_dir,
    min_rows=2,
    max_cols=4,
    max_rows=4,
):
    """Plot images rotated with predicted angles
    Args:
        input_images (torch tensor): shape (bs, c, h, w)
                                     images
        coords_pred (torch tensor): shape (bs, 5)
                     each row is [xc, yc, xt, yt, w] is prediction
        theta_pred (torch tensor): shape (bs) with predictions of theta
                                   in radians
        prefix (string): prefix for plot filename
        output_dir (string): path to output directory to save plot
        min_rows (int): min number of rows in plot (default=2)
        max_cols (int): max number of columns in plot (default=4)
        max_rows (int): max number of rows in plot (default=4)
    """
    images_un = unnormalize(input_images).numpy().transpose(0, 2, 3, 1)
    coords_pred = coords_pred.numpy()
    bs = input_images.size(0)
    ncols = min(max_cols, bs)
    nrows = int(math.ceil(float(bs) / ncols))
    if nrows == 1:
        nrows = 2
        ncols = bs // nrows
    nrows = min(nrows, max_rows)

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 8, nrows * 4))
    for r in range(nrows):
        for c in range(ncols):
            # If no images left in a batch, do not plot
            if r * ncols + c >= bs:
                continue

            # Plot predictions
            degrees_pred = math.degrees(theta_pred[r * ncols + c])
            image_rotated = transform.rotate(
                images_un[r * ncols + c],
                angle=degrees_pred,
                center=coords_pred[r * ncols + c, :1],
            )
            image_combined = np.concatenate(
                [images_un[r * ncols + c], image_rotated], axis=1
            )
            ax[r, c].imshow(image_combined)
            ax[r, c].set_title('Rotated by {:.0f} deg'.format(degrees_pred))
            ax[r, c].axis('off')

    # Save plot
    file_name = os.path.join(output_dir, 'debug_images', '{}.png'.format(prefix))
    fig.savefig(file_name, format='png', dpi=100, bbox_inches='tight', facecolor='w')
    plt.close(fig)


def plot_theta_err_hist(theta_gt, theta_pred, prefix, output_dir):
    """
    Plot histogram of errors of angle theta predictions
    Args:
        theta_gt (torch tensor): shape (bs) with ground truth values of theta
                                 in radians
        theta_pred (torch tensor): shape (bs) with predictions of theta
                                   in radians
        prefix (string): prefix for plot filename
        output_dir (string): path to output directory to save plot
    """
    style.use('seaborn')
    np_norm_theta = np.vectorize(normalize_theta)
    theta_pred = np.rad2deg(theta_pred)
    theta_gt = np.rad2deg(theta_gt)
    err_theta = np.abs(np_norm_theta(theta_pred) - np_norm_theta(theta_gt))
    err_theta = np.abs(np_norm_theta(err_theta))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(err_theta, bins=36)
    ax.set_xticks(list(range(0, 180, 10)))
    ax.set_xlabel('Error in degrees')
    ax.set_ylabel('Number of images')

    # Save plot
    file_name = os.path.join(output_dir, 'hist_{}.png'.format(prefix))
    fig.savefig(file_name, format='png', dpi=100, bbox_inches='tight', facecolor='w')
    plt.close(fig)
    style.use('classic')
