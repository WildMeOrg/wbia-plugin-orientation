# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# Written by Olga Moskvyak (olga.moskvyak@hdr.qut.edu.au)
# ------------------------------------------------------------------------------

import math
import os
import matplotlib.pyplot as plt

from utils.utils import unnormalize
from utils.data_manipulation import plot_image_coordinates


def save_batch_images(batch_image, coords, theta, file_name, min_rows=2, max_cols=8):
    '''
    batch_image: torch tensor of shape (bs, c, h, w)
    coords: torch tensor shape (bs, 5) where each row is [xc, yc, xt, yt, w]
    theta: torch tensor shape (bs) with values of theta
    file_name: string, name of the file to save plot
    max_cols: int, max number of columns in plot
    '''
    images_un = unnormalize(batch_image)
    bs = batch_image.size(0)
    ncols = min(max_cols, bs)
    nrows = int(math.ceil(float(bs) / ncols))
    if nrows == 1:
        nrows = 2
        ncols = bs // nrows
        
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*4, nrows*4))
    for r in range(nrows):
        for c in range(ncols):
            plot_image_coordinates(ax[r,c], 
                                   images_un[r*ncols+c].numpy().transpose((1,2,0)), 
                                   coords[r*ncols+c, 0].numpy(), 
                                   coords[r*ncols+c, 1].numpy(), 
                                   coords[r*ncols+c, 2].numpy(), 
                                   coords[r*ncols+c, 3].numpy(), 
                                   coords[r*ncols+c, 4].numpy())
            if theta is not None:
                ax[r,c].set_title('Theta {:.2f}'.format(math.degrees(theta[r*ncols+c])))
    fig.savefig(file_name, format='png', dpi=100, bbox_inches='tight', facecolor='w')

def save_debug_images(config, input_images, coords_gt, coords_pred, theta_gt, theta_preds, prefix, output_dir):
    if not config.DEBUG.DEBUG:
        return

    if config.DEBUG.SAVE_BATCH_IMAGES_GT:
        save_batch_images(input_images, coords_gt, theta_gt, 
                          os.path.join(output_dir, 'debug_images', '{}_gt.png'.format(prefix)))
    if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
        save_batch_images(input_images, coords_pred, theta_preds,
                          os.path.join(output_dir, 'debug_images', '{}_pred.png'.format(prefix)))       