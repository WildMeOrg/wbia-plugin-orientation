# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from wbia.control import controller_inject
import numpy as np
import utool as ut
import wbia
import os

# import sys
import torch
import json
import matplotlib.pyplot as plt
from skimage import transform
import math
import random

import torchvision.transforms as transforms  # noqa: E402

from wbia_orientation.config.default import _C as cfg  # noqa
from wbia_orientation.core.evaluate import compute_theta  # noqa: E402
from wbia_orientation.dataset.animal_wbia import AnimalWbiaDataset  # noqa: E402
from wbia_orientation.train import _make_model, _model_to_gpu  # noqa: E402
from wbia_orientation.utils.data_manipulation import resize_oa_box  # noqa: E402
from wbia_orientation.utils.data_manipulation import plot_image_bbox  # noqa: E402
from wbia_orientation.utils.data_manipulation import plot_image_coordinates  # noqa: E402
from wbia_orientation.utils.file_downloader import download_file  # noqa: E402

(print, rrr, profile) = ut.inject2(__name__)

_, register_ibs_method = controller_inject.make_ibs_register_decorator(__name__)

register_api = controller_inject.get_wbia_flask_api(__name__)
register_route = controller_inject.get_wbia_flask_route(__name__)


PROJECT_PATH = os.path.abspath(os.path.join(os.path.split(__file__)[0], '..'))


MODEL_URLS = {
    'seaturtle': 'https://wildbookiarepository.azureedge.net/models/orientation.seaturtle.20201120.pth',
    'seadragon': 'https://wildbookiarepository.azureedge.net/models/orientation.seadragon.20201120.pth',
    'whaleshark': 'https://wildbookiarepository.azureedge.net/models/orientation.whaleshark.20201120.pth',
    'mantaray': 'https://wildbookiarepository.azureedge.net/models/orientation.mantaray.20201120.pth',
    'spotteddolphin': 'https://wildbookiarepository.azureedge.net/models/orientation.spotteddolphin.20201120.pth',
    'hammerhead': 'https://wildbookiarepository.azureedge.net/models/orientation.hammerhead.20201120.pth',
    'rightwhale': 'https://wildbookiarepository.azureedge.net/models/orientation.rightwhale.20201120.pth',
}

CONFIGS = {
    'seaturtle': 'https://wildbookiarepository.azureedge.net/models/orientation.seaturtle.20201120.yaml',
    'seadragon': 'https://wildbookiarepository.azureedge.net/models/orientation.seadragon.20201120.yaml',
    'whaleshark': 'https://wildbookiarepository.azureedge.net/models/orientation.whaleshark.20201120.yaml',
    'mantaray': 'https://wildbookiarepository.azureedge.net/models/orientation.mantaray.20201120.yaml',
    'spotteddolphin': 'https://wildbookiarepository.azureedge.net/models/orientation.spotteddolphin.20201120.yaml',
    'hammerhead': 'https://wildbookiarepository.azureedge.net/models/orientation.hammerhead.20201120.yaml',
    'rightwhale': 'https://wildbookiarepository.azureedge.net/models/orientation.rightwhale.20201120.yaml',
}

DATA_ARCHIVES = {
    'seaturtle': 'https://wildbookiarepository.azureedge.net/datasets/orientation.seaturtle.coco.tar.gz',
    'seadragon': 'https://wildbookiarepository.azureedge.net/datasets/orientation.seadragon.coco.tar.gz',
    'whaleshark': 'https://wildbookiarepository.azureedge.net/datasets/orientation.whaleshark.coco.tar.gz',
    'mantaray': 'https://wildbookiarepository.azureedge.net/datasets/orientation.mantaray.coco.tar.gz',
    'spotteddolphin': 'https://wildbookiarepository.azureedge.net/datasets/orientation.spotteddolphin.coco.tar.gz',
    'hammerhead': 'https://wildbookiarepository.azureedge.net/datasets/orientation.hammerhead.coco.tar.gz',
    'rightwhale': 'https://wildbookiarepository.azureedge.net/datasets/orientation.rightwhale.coco.tar.gz',
}

register_preproc_image = controller_inject.register_preprocs['image']
register_preproc_annot = controller_inject.register_preprocs['annot']
# register_preproc_part  = controller_inject.register_preprocs['part']


@register_ibs_method
@register_api('/api/plugin/orientation/', methods=['GET', 'POST'])
def wbia_plugin_detect_oriented_box(
    ibs, aid_list, species, use_gpu=False, plot_samples=True
):
    r"""
    Detect orientation of animals in images

    Args:
        ibs (WBIAController):  wbia controller object
        aid_list (int): annot ids specifying the input
        species (string): type of species
        use_gpu (bool): use GPU or CPU for model inference (default: False)
        plot_samples (bool): plot some samples and save to disk (default: True)

    Returns:
        list of lists: list of params of object-oriented boxes
            [[xc, yc, xt, yt, w], ...]
        list of floats: list of angles of rotation in radians

    CommandLine:
        python -m wbia_orientation._plugin --test-wbia_plugin_detect_oriented_box

    Example:
        >>> # ENABLE_DOCTEST
        >>> import wbia
        >>> import wbia_orientation
        >>> species = 'spotteddolphin'
        >>> url = 'https://wildbookiarepository.azureedge.net/datasets/orientation.spotteddolphin.coco.tar.gz'
        >>> ibs = wbia_orientation._plugin.wbia_orientation_test_ibs(species, dataset_url=url)
        >>> aid_list = ibs.get_valid_aids()
        >>> aid_list = aid_list[:10]
        >>> output, theta = ibs.wbia_plugin_detect_oriented_box(aid_list, species, False, False)
        >>> expected_theta = [-0.3071622848510742, 1.2332571744918823,
                              1.6512340307235718, 1.6928660869598389,
                              1.3716390132904053, 4.61941385269165,
                              1.1511050462722778, 1.093467116355896,
                              1.1569938659667969, 0.7397593855857849]
        >>> import numpy as np
        >>> diff = np.abs(np.array(theta) - np.array(expected_theta))
        >>> assert diff.all() < 1e-6

    Example:
        >>> # ENABLE_DOCTEST
        >>> import wbia
        >>> import wbia_orientation
        >>> species = 'seadragon'
        >>> select_cats = [1,3]
        >>> url = 'https://wildbookiarepository.azureedge.net/datasets/orientation.seadragon.coco.tar.gz'
        >>> ibs = wbia_orientation._plugin.wbia_orientation_test_ibs(species, select_cats=select_cats, dataset_url=url)
        >>> aid_list = ibs.get_valid_aids()
        >>> aid_list = aid_list[:10]
        >>> output, theta = ibs.wbia_plugin_detect_oriented_box(aid_list, species, False, False)
        >>> expected_theta = [2.2275471687316895, 4.496161937713623,
                              3.693049430847168, 3.4513893127441406,
                              3.496103525161743, 4.1899213790893555,
                              4.020716190338135, 2.2543320655822754,
                              3.9189162254333496, 2.3440582752227783]
        >>> import numpy as np
        >>> diff = np.abs(np.array(theta) - np.array(expected_theta))
        >>> assert diff.all() < 1e-6
    """
    assert (
        species in CONFIGS.keys()
    ), 'Species {} is not supported. The following \
        species are supported: {}'.format(
        species, list(CONFIGS.keys())
    )

    # A. Load config and model
    cfg = _load_config(species, use_gpu)
    model = _load_model(cfg, MODEL_URLS[species])

    # B. Preprocess image to model input
    test_loader, test_dataset, bboxes = orientation_load_data(
        ibs, aid_list, cfg.MODEL.IMSIZE, cfg
    )

    # C. Compute output
    outputs = []
    model.eval()
    with torch.no_grad():
        for images in test_loader:
            if cfg.USE_GPU:
                images = images.cuda(non_blocking=True)

            # Compute output of Orientation Network
            output = model(images.float(), cfg.TEST.HFLIP, cfg.TEST.VFLIP, cfg.USE_GPU)
            outputs.append(output)

    # Post-processing
    outputs, theta = orientation_post_proc(outputs, bboxes)

    # Plot random samples
    if plot_samples:
        wbia_orientation_plot(
            ibs,
            aid_list,
            bboxes,
            outputs,
            theta,
            species,
            nrows=3,
            ncols=4,
        )

    return outputs, theta


def orientation_load_data(ibs, aid_list, target_imsize, cfg):
    r"""
    Load data, preprocess and create data loaders
    """
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image_paths = ibs.get_annot_image_paths(aid_list)
    bboxes = ibs.get_annot_bboxes(aid_list)
    test_dataset = AnimalWbiaDataset(image_paths, bboxes, target_imsize, test_transform)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.TEST.BS * len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        drop_last=False,
    )
    print('Loaded {} samples'.format(len(test_dataset)))

    return test_loader, test_dataset, bboxes


def orientation_post_proc(output, bboxes):
    r"""
    Post-process model output to get object-oriented bounding box
    angles of rotation theta in the original image

    Args:
        output (torch tensor): tensor of shape (bs, 5), output of the model
                               each row is [xc, yc, xt, yt, w],
                               coords are between 0 and 1
       bboxes (list of tuples): list of bounding box coordinates (x, y, w, h)

    Returns:
        output (list of lists): list of sublists [xc, yc, xt, yt, w]
                                coordinates are absolute in the original image
        theta (list of floats): list of angles of rotation in radians
    """
    # Concatenate and convert to numpy
    output = torch.cat(output, dim=0).numpy()

    # Compute theta from coordinates of object-aligned box
    theta = compute_theta(output)

    # Resize coords back to original size
    for i in range(len(output)):
        # Each row in output is an array of 5 [xc, yc, xt, yt, w]
        # Bboxes is of format (x, y, w, h) while target size is [h, w]
        x1, y1, bw, bh = bboxes[i]
        output[i] = resize_oa_box(
            output[i], original_size=[1.0, 1.0], target_size=[bh, bw]
        )

        # Shift coordinates from bounding box origin to original origin
        output[i][0] += x1
        output[i][1] += y1
        output[i][2] += x1
        output[i][3] += y1

    # Convert to lists
    output = output.tolist()
    theta = theta.tolist()
    return output, theta


def _load_config(species, use_gpu):
    r"""
    Load a configuration file for species
    """
    config_url = CONFIGS[species]

    config_fname = config_url.split('/')[-1]
    config_file = ut.grab_file_url(
        config_url, appname='wbia_orientation', check_hash=True, fname=config_fname
    )

    cfg.defrost()
    cfg.merge_from_file(config_file)
    cfg.USE_GPU = use_gpu
    cfg.freeze()
    return cfg


def _load_model(cfg, model_url=None):
    r"""
    Load a model based on config file
    """
    model = _make_model(cfg, is_train=False)

    # Download the model and put it in the models folder
    if model_url is not None:
        # os.makedirs('models', exist_ok=True)  # Note: Use system-specific cache folder
        model_fname = model_url.split('/')[-1]
        model_path = ut.grab_file_url(
            model_url, appname='wbia_orientation', check_hash=True, fname=model_fname
        )
    else:
        model_path = cfg.TEST.MODEL_FILE

    import torch

    if cfg.USE_GPU:
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    print('Loaded model from {}'.format(model_path))
    model = _model_to_gpu(model, cfg)
    return model


def wbia_orientation_test_ibs(
    species,
    subset='test2020',
    select_cats=[],
    data_dir='data/downloaded_annot_archives',
    dataset_url='',
):
    r"""
    Create a database to test orientation detection from a coco annotation file
    """
    assert (
        species in CONFIGS.keys()
    ), 'Species {} is not supported. The following \
        species are supported: {}'.format(
        species, list(CONFIGS.keys())
    )

    testdb_name = os.path.join('data', 'testdb_' + species + '_' + subset)
    test_ibs = wbia.opendb(testdb_name, allow_newdir=True)
    if len(test_ibs.get_valid_aids()) > 0:
        return test_ibs
    else:
        # Download data archive
        download_file(dataset_url, data_dir, extract=True)
        # Load coco annotations
        db_coco_ann_path = os.path.join(data_dir, 'orientation.{}.coco'.format(species))
        test_annots = os.path.join(
            db_coco_ann_path, 'annotations', 'instances_{}.json'.format(subset)
        )
        dataset = json.load(open(test_annots, 'r'))

        # Get image paths and add them to the database
        impaths = [
            os.path.join(db_coco_ann_path, 'images', subset, d['file_name'])
            for d in dataset['images']
        ]
        gid_list = test_ibs.add_images(impaths)

        # Get annotations and add them to the database
        gid_annots = []
        bbox_list = []
        theta_list = []
        imageid2filename = {d['id']: d['file_name'] for d in dataset['images']}
        filenames = [d['file_name'] for d in dataset['images']]

        annotations = dataset['annotations']
        if 'parts' in dataset:
            annotations += dataset['parts']

        for ann in annotations:
            # Select annotation.
            # The annotation is included if:
            # a. there is no list of selected categories in config
            # b. object category in the list of selected categories in config
            if len(select_cats) == 0 or (
                len(select_cats) > 0 and ann['category_id'] in select_cats
            ):
                gid_annots.append(
                    gid_list[filenames.index(imageid2filename[ann['image_id']])]
                )
                bbox_list.append(ann['segmentation_bbox'])
                theta_list.append(ann['theta'])

        test_ibs.add_annots(gid_annots, bbox_list=bbox_list, theta_list=theta_list)

        return test_ibs


def wbia_orientation_plot(
    ibs, aid_list, bboxes, output, theta, prefix, output_dir=None, nrows=4, ncols=4
):
    r"""
    Plot random examples
    """
    if output_dir is None:
        output_dir = os.path.abspath(os.path.join(PROJECT_PATH, 'examples'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 4))
    idx_plot = random.sample(
        list(range(len(aid_list))), min(nrows * ncols, len(aid_list))
    )
    idx_plot = list(range(len(aid_list)))
    aid_plot = [aid_list[i] for i in idx_plot]
    bboxes_plot = [bboxes[i] for i in idx_plot]
    output_plot = [output[i] for i in idx_plot]
    theta_plot = [theta[i] for i in idx_plot]

    for r in range(nrows):
        for c in range(ncols):
            # If no images left, do not plot
            if r * ncols + c >= len(aid_list):
                continue
            aid = aid_plot[r * ncols + c]
            image_original = ibs.get_annot_images([aid])[0]

            # Plot original box
            plot_image_bbox(
                ax[r, c], image_original[:, :, ::-1], bboxes_plot[r * ncols + c]
            )

            # Plot corrected object-aligned box
            xc, yc, xt, yt, w = output_plot[r * ncols + c]
            plot_image_coordinates(
                ax[r, c],
                image_original[:, :, ::-1],
                xc,
                yc,
                xt,
                yt,
                w,
                marker='r-',
            )
            ax[r, c].axis('off')

    plt.tight_layout()
    # Save plot
    file_name = os.path.join(output_dir, '{}_bboxes.jpg'.format(prefix))
    fig.savefig(file_name, format='jpg', bbox_inches='tight', facecolor='w')
    plt.close(fig)

    fig = plt.figure(constrained_layout=False, figsize=(ncols * 8, nrows * 4))
    outer_gs = fig.add_gridspec(nrows=nrows, ncols=ncols)

    for r in range(nrows):
        for c in range(ncols):
            # Set grid spec
            gs = outer_gs[r, c].subgridspec(1, 2, wspace=0.0)
            left = fig.add_subplot(gs[0])
            right = fig.add_subplot(gs[1])

            # If no images left, do not plot
            if r * ncols + c >= len(aid_list):
                continue

            aid = aid_plot[r * ncols + c]
            x1, y1, bw, bh = bboxes_plot[r * ncols + c]
            image_original = ibs.get_annot_images([aid])[0]
            # Crop patch and rotate image
            image_bbox = image_original[y1 : y1 + bh, x1 : x1 + bw]
            left.imshow(image_bbox[:, :, ::-1])
            left.set_title('Input', fontsize=14)
            left.axis('off')

            # Compute theta from predicted coordinates
            xc, yc, xt, yt, w = output_plot[r * ncols + c]
            xc -= x1
            yc -= y1
            theta_degree = math.degrees(theta_plot[r * ncols + c])

            # Rotate cropped patch
            rotation_centre = np.asarray([xc, yc])
            image_bbox_rot = transform.rotate(
                image_bbox,
                angle=theta_degree,
                center=rotation_centre,
                mode='constant',
                resize=True,
                preserve_range=True,
            ).astype('uint8')
            right.imshow(image_bbox_rot[:, :, ::-1])
            right.set_title('Rotated by {:.0f} degrees'.format(theta_degree), fontsize=14)
            right.axis('off')

    # Save plot
    file_name = os.path.join(output_dir, '{}_rotated.jpg'.format(prefix))
    fig.savefig(file_name, format='jpg', bbox_inches='tight', facecolor='w')
    plt.close(fig)


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia_orientation._plugin --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
