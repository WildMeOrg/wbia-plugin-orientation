# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import itertools
import numpy as np
import torchvision
import cv2
import matplotlib.pyplot as plt

from utils.utils import unnormalize


def save_batch_image_with_joints(batch_image, batch_joints, batch_joints_vis,
                                 file_name, nrow=8, padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]

            for joint, joint_vis in zip(joints, joints_vis):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                if joint_vis[0]:
                    cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2, [255, 0, 0], 2)
            k = k + 1
    cv2.imwrite(file_name, ndarr)


def save_batch_heatmaps(batch_image, batch_heatmaps, file_name,
                        normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros((batch_size*heatmap_height,
                           (num_joints+1)*heatmap_width,
                           3),
                          dtype=np.uint8)

    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            cv2.circle(resized_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap*0.7 + resized_image*0.3
            cv2.circle(masked_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j+1)
            width_end = heatmap_width * (j+2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)


def save_debug_images(config, input_images, coords_gt, coords_vis_gt, target, joints_pred, output,
                      prefix):
    if not config.DEBUG.DEBUG:
        return

    if config.DEBUG.SAVE_BATCH_IMAGES_GT:
        save_batch_image_with_joints(
            input_images, coords_gt, coords_vis_gt,
            '{}_gt.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
        save_batch_image_with_joints(
            input_images, joints_pred, coords_vis_gt,
            '{}_pred.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_HEATMAPS_GT:
        save_batch_heatmaps(
            input_images, target, '{}_hm_gt.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_HEATMAPS_PRED:
        save_batch_heatmaps(
            input_images, output, '{}_hm_pred.jpg'.format(prefix)
        )
        
def save_reconstructed(config, input_images, input_images_aug, image_rec, image_aug_rec, output_dir, iteration):
    """Save torch tensor of input and reconstructed images
    """
    output_dir = os.path.join(output_dir, 'reconstructed')
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    input_images = unnormalize(input_images, use_gpu=config.USE_GPU)  
    input_images_aug = unnormalize(input_images_aug, use_gpu=config.USE_GPU)          
    
    input_images = np.clip(np.transpose(input_images.cpu().numpy(), (0, 2, 3, 1)), 0., 1.)
    input_images_aug = np.clip(np.transpose(input_images_aug.cpu().numpy(), (0, 2, 3, 1)), 0., 1.)
    image_rec = np.clip(np.transpose(image_rec.detach().cpu().numpy(), (0, 2, 3, 1)), 0., 1.)
    image_aug_rec = np.clip(np.transpose(image_aug_rec.detach().cpu().numpy(), (0, 2, 3, 1)), 0., 1.)
    
    print_image_grid([input_images[:8], input_images_aug[:8], image_rec[:8], image_aug_rec[:8]], showFig=False, 
                     saveFig=True, 
                     figName=os.path.join(output_dir, 'iteration_{}.png'.format(iteration)), 
                     titles=['Input', 'Input-Augmented', 'Rec Input-col 1', 'Rec Input-Aug-col 2'])
    
def print_image_grid(imgs, showFig=True, saveFig=False, figName='swapped.png', titles=None):
    '''Print image grid
    imgs: list of numpy arrays or lists. 
            all elements should have the same length. Sublists are be printed as columns
    titles: list of strings. title for each column
    '''
    ncols = len(imgs)
    for i in range(ncols-1):
        assert len(imgs[i]) == len(imgs[i+1])
    
    nrows = len(imgs[0])
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3*ncols, 2*nrows))

    if titles:
        for i in range(ncols):
            ax[0, i].set_title(titles[i])
    
    for row_id in range(nrows):
        for col_id in range(ncols):
            ax[row_id, col_id].imshow(np.squeeze(imgs[col_id][row_id]))
    
    for i in itertools.product(range(nrows), range(ncols)):
        ax[i].axis('off')
        
    plt.tight_layout()
    if showFig:
        plt.show()
    if saveFig:
        fig.savefig(figName)
    plt.close(fig)
            
