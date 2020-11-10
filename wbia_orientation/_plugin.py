# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from wbia.control import controller_inject
import numpy as np
import utool as ut
import wbia
import os
import sys
import torch
import json
import matplotlib.pyplot as plt
from skimage import transform
import math
import random

this_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(this_dir, '..', 'wbia_orientation'))

from dataset.animal_wbia import AnimalWbiaDataset  # noqa: E402
import torchvision.transforms as transforms  # noqa: E402
from config.default import _C as cfg  # noqa: E402
from train import _make_model, _model_to_gpu  # noqa: E402
from utils.data_manipulation import resize_oa_box  # noqa: E402
from core.evaluate import compute_theta  # noqa: E402
from utils.data_manipulation import plot_image_bbox  # noqa: E402
from utils.data_manipulation import plot_image_coordinates  # noqa: E402

(print, rrr, profile) = ut.inject2(__name__)

_, register_ibs_method = controller_inject.make_ibs_register_decorator(__name__)

register_api = controller_inject.get_wbia_flask_api(__name__)
register_route = controller_inject.get_wbia_flask_route(__name__)

# TODO upload models
MODEL_URLS = {
    'seaturtle': 'https://wildbookiarepository.azureedge.net/models/orientation.seaturtle.pth',
    'seadragon': 'https://wildbookiarepository.azureedge.net/models/orientation.seadragib.pth',
    'whaleshark': 'https://wildbookiarepository.azureedge.net/models/orientation.whaleshark.pth',
    'mantaray': 'https://wildbookiarepository.azureedge.net/models/orientation.mantaray.pth',
    'spotteddolphin': 'https://wildbookiarepository.azureedge.net/models/orientation.spotteddolphin.pth',
    'hammerhead': 'https://wildbookiarepository.azureedge.net/models/orientation.hammerhead.pth',
    'rightwhale': 'https://wildbookiarepository.azureedge.net/models/orientation.rightwhale.pth',
}

CONFIGS = {
    'seaturtle': 'wbia_orientation/config/seaturtle.yaml',
    'seadragon': 'wbia_orientation/config/seadragon.yaml',
    'whaleshark': 'wbia_orientation/config/whaleshark.yaml',
    'mantaray': 'wbia_orientation/config/mantaray.yaml',
    'spotteddolphin': 'wbia_orientation/config/spotteddolphin.yaml',
    'hammerhead': 'wbia_orientation/config/hammerhead.yaml',
    'rightwhale': 'wbia_orientation/config/rightwhale.yaml',
}

register_preproc_image = controller_inject.register_preprocs['image']
register_preproc_annot = controller_inject.register_preprocs['annot']
# register_preproc_part  = controller_inject.register_preprocs['part']


@register_ibs_method
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
        >>> ibs = wbia_orientation._plugin.wbia_orientation_test_ibs(species)
        >>> aid_list = ibs.get_valid_aids()
        >>> aid_list = aid_list[:10]
        >>> output, theta = ibs.wbia_plugin_detect_oriented_box(aid_list, species, False, False)
        >>> expected_theta = [-0.4158303737640381, 1.5231519937515259,
                              2.0344438552856445, 1.6124389171600342,
                              1.5768203735351562, 4.669830322265625,
                              1.3162155151367188, 1.2578175067901611,
                              0.9936041831970215,  0.8561460971832275]
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
    model = _load_model(cfg, cfg.TEST.MODEL_FILE)

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
            output_dir='./examples',
            nrows=4,
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
    Post-process model output to get params of object-oriented bounding box
    in the original image and angles of rotation theta

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

    # Resize coords back to original size
    for i in range(len(output)):
        # Each row in output is an array of 5 [xc, yc, xt, yt, w]
        # Bboxes is of format (x, y, w, h) while target size is [h, w]
        output[i] = resize_oa_box(
            output[i], original_size=[1.0, 1.0], target_size=[bboxes[i][3], bboxes[i][2]]
        )

    # Compute theta from coordinates of object-aligned box
    theta = compute_theta(output)

    # Convert to lists
    output = output.tolist()
    theta = theta.tolist()
    return output, theta


def _load_config(species, use_gpu):
    r"""
    Load a configuration file for species
    """
    config_file = CONFIGS[species]
    cfg.defrost()
    cfg.merge_from_file(config_file)
    cfg.USE_GPU = use_gpu
    cfg.freeze()
    return cfg


def _load_model(cfg, model_path):
    r"""
    Load a model based on config file
    """
    model = _make_model(cfg, is_train=False)

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
    coco_annot_dir='/external/contractors/olga.moskvyak/data',
    subset='test2020',
    select_cats=[],
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

    testdb_name = 'testdb_' + species
    test_ibs = wbia.opendb(testdb_name, allow_newdir=True)
    if len(test_ibs.get_valid_gids()) > 0:
        return test_ibs
    else:
        # Load coco annotations
        db_coco_ann_path = os.path.join(
            coco_annot_dir, 'orientation.{}.coco'.format(species)
        )
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

        for ann in dataset['annotations']:
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
    ibs, aid_list, bboxes, output, theta, prefix, output_dir='./', nrows=4, ncols=4
):
    r"""
    Plot random examples
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 4))
    idx_plot = random.sample(
        list(range(len(aid_list))), min(nrows * ncols, len(aid_list))
    )
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
            x1, y1, bw, bh = bboxes_plot[r * ncols + c]
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
                xc + x1,
                yc + y1,
                xt + x1,
                yt + y1,
                w,
                marker='r-',
            )
            ax[r, c].axis('off')

    plt.tight_layout()
    # Save plot
    file_name = os.path.join(output_dir, '{}_random_bboxes.png'.format(prefix))
    fig.savefig(file_name, format='png', dpi=100, bbox_inches='tight', facecolor='w')
    plt.close(fig)

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 8, nrows * 4))

    for r in range(nrows):
        for c in range(ncols):
            # If no images left, do not plot
            if r * ncols + c >= len(aid_list):
                continue
            aid = aid_plot[r * ncols + c]
            x1, y1, bw, bh = bboxes_plot[r * ncols + c]
            image_original = ibs.get_annot_images([aid])[0]
            # Crop patch and rotate image
            image_bbox = image_original[y1 : y1 + bh, x1 : x1 + bw]

            # Compute theta from predicted coordinates
            xc, yc, xt, yt, w = output_plot[r * ncols + c]
            theta_degree = math.degrees(theta_plot[r * ncols + c])
            image_bbox_rot = transform.rotate(
                image_bbox,
                angle=theta_degree,
                center=[xc, yc],
                preserve_range=True,
            ).astype('uint8')

            # Plot side-by-side patches cropped by original axis-aligned bbox
            # and detected object-aligned box
            image_plot = np.concatenate(
                [image_bbox[:, :, ::-1], image_bbox_rot[:, :, ::-1]], axis=1
            )
            ax[r, c].imshow(image_plot)
            ax[r, c].set_title('Rotated by {} degrees'.format(theta_degree))
            ax[r, c].axis('off')
    plt.tight_layout()
    # Save plot
    file_name = os.path.join(output_dir, '{}_random_rotated.png'.format(prefix))
    fig.savefig(file_name, format='png', dpi=100, bbox_inches='tight', facecolor='w')
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
