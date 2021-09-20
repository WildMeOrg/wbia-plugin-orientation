# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from wbia.control import controller_inject
import numpy as np
import utool as ut
import wbia
import os

from wbia import plottool as pt

# import sys
import torch
import json
import matplotlib.pyplot as plt
from skimage import transform
import math
import random
import tqdm

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
    'beluga_whale_v0': 'https://wildbookiarepository.azureedge.net/models/orientation.beluga.20210429.pth',
    'whale_sperm_v0': 'https://wildbookiarepository.azureedge.net/models/orientation.whale_sperm.v0.pth',
}

CONFIGS = {
    'seaturtle': 'https://wildbookiarepository.azureedge.net/models/orientation.seaturtle.20201120.yaml',
    'seadragon': 'https://wildbookiarepository.azureedge.net/models/orientation.seadragon.20201120.yaml',
    'whaleshark': 'https://wildbookiarepository.azureedge.net/models/orientation.whaleshark.20201120.yaml',
    'mantaray': 'https://wildbookiarepository.azureedge.net/models/orientation.mantaray.20201120.yaml',
    'spotteddolphin': 'https://wildbookiarepository.azureedge.net/models/orientation.spotteddolphin.20201120.yaml',
    'hammerhead': 'https://wildbookiarepository.azureedge.net/models/orientation.hammerhead.20201120.yaml',
    'rightwhale': 'https://wildbookiarepository.azureedge.net/models/orientation.rightwhale.20201120.yaml',
    'beluga_whale_v0': 'https://wildbookiarepository.azureedge.net/models/orientation.beluga.20210429.yaml',
    'whale_sperm_v0': 'https://wildbookiarepository.azureedge.net/models/orientation.whale_sperm.v0.yaml',
}

DATA_ARCHIVES = {
    'seaturtle': 'https://wildbookiarepository.azureedge.net/datasets/orientation.seaturtle.coco.tar.gz',
    'seadragon': 'https://wildbookiarepository.azureedge.net/datasets/orientation.seadragon.coco.tar.gz',
    'whaleshark': 'https://wildbookiarepository.azureedge.net/datasets/orientation.whaleshark.coco.tar.gz',
    'mantaray': 'https://wildbookiarepository.azureedge.net/datasets/orientation.mantaray.coco.tar.gz',
    'spotteddolphin': 'https://wildbookiarepository.azureedge.net/datasets/orientation.spotteddolphin.coco.tar.gz',
    'hammerhead': 'https://wildbookiarepository.azureedge.net/datasets/orientation.hammerhead.coco.tar.gz',
    'rightwhale': 'https://wildbookiarepository.azureedge.net/datasets/orientation.rightwhale.coco.tar.gz',
    'beluga_whale_v0': None,
    'whale_sperm_v0': None,
}

SPECIES_MODEL_TAG_MAPPING = {
    'right_whale_head': 'rightwhale',
    'turtle_green+head': 'seaturtle',
    'turtle_hawksbill+head': 'seaturtle',
    'turtle_oliveridley+head': 'seaturtle',
    'turtle_sea+head': 'seaturtle',
    'seadragon_leafy+head': 'seadragon',
    'seadragon_weedy+head': 'seadragon',
    'shark_hammerhead': 'hammerhead',
    'manta_ray': 'mantaray',
    'manta_ray_giant': 'mantaray',
    'mobula_birostris': 'mantaray',
    'dolphin_spotted': 'spotteddolphin',
    'whale_shark': 'whaleshark',
    'whale_beluga': 'beluga_whale_v0',
    'whale_sperm': 'whale_sperm_v0',
    'whale_sperm+fluke': 'whale_sperm_v0',
    'physeter_macrocephalus': 'whale_sperm_v0',
    'physeter_macrocephalus+fluke': 'whale_sperm_v0',
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


@register_ibs_method
def wbia_plugin_orientation_render_examples(
    ibs, num_examples=10, desired_note='random-01', **kwargs
):
    r"""
    Show examples of the prediction for each species

    Args:
        ibs       (IBEISController): IBEIS controller object
        aid_list  (list of int): A list of IBEIS Annotation IDs (aids)
        model_tag (string): Key to URL_DICT entry for this model

    Returns:
        theta_list

    CommandLine:
        python -m wbia_orientation._plugin --test-wbia_plugin_orientation_render_examples

    Example0:
        >>> # ENABLE_DOCTEST
        >>> import wbia
        >>> import random
        >>> from wbia.init import sysres
        >>> import numpy as np
        >>> dbdir = sysres.ensure_testdb_orientation()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> fig_filepath = ibs.wbia_plugin_orientation_render_examples(desired_note='source')
        >>> fig_filepath = ibs.wbia_plugin_orientation_render_examples(desired_note='aligned')
        >>> fig_filepath = ibs.wbia_plugin_orientation_render_examples(desired_note='random-01')
        >>> print(fig_filepath)
    """
    INPUT_SIZE = 224 * 2

    import random

    random.seed(1)

    aid_list = ibs.get_valid_aids()
    note_list = ibs.get_annot_notes(aid_list)
    note_list = np.array(note_list)
    flag_list = note_list == desired_note

    aid_list = ut.compress(aid_list, flag_list)
    species_list = ibs.get_annot_species(aid_list)
    species_list = np.array(species_list)

    key_list = [
        'mobula_birostris',
        'right_whale_head',
        'seadragon_weedy+head',
        'turtle_hawksbill+head',
    ]

    all_aid_list = []
    oriented_aid_list = []
    result_dict = {}

    for key in key_list:
        model_tag = SPECIES_MODEL_TAG_MAPPING.get(key, None)
        assert model_tag is not None

        flag_list = species_list == key
        aid_list_ = ut.compress(aid_list, flag_list)
        random.shuffle(aid_list_)
        aid_list_ = aid_list_[:num_examples]

        config = {
            'orienter_algo': 'plugin:orientation',
            'orienter_weight_filepath': None,
        }
        result_list = ibs.depc_annot.get('orienter', aid_list_, None, config=config)

        xtl_list = list(map(int, map(np.around, ut.take_column(result_list, 0))))
        ytl_list = list(map(int, map(np.around, ut.take_column(result_list, 1))))
        w_list = list(map(int, map(np.around, ut.take_column(result_list, 2))))
        h_list = list(map(int, map(np.around, ut.take_column(result_list, 3))))
        theta_list_ = ut.take_column(result_list, 4)
        bbox_list_ = list(zip(xtl_list, ytl_list, w_list, h_list))

        gid_list_ = ibs.get_annot_gids(aid_list_)
        species_list_ = ibs.get_annot_species(aid_list_)
        viewpoint_list_ = ibs.get_annot_viewpoints(aid_list_)
        name_list_ = ibs.get_annot_names(aid_list_)
        note_list_ = ['TEMPORARY'] * len(aid_list_)

        oriented_aid_list_ = ibs.add_annots(
            gid_list_,
            bbox_list=bbox_list_,
            theta_list=theta_list_,
            species_list=species_list_,
            viewpoint_list=viewpoint_list_,
            name_list=name_list_,
            notes_list=note_list_,
        )
        oriented_aid_list += oriented_aid_list_

        result_dict[key] = list(zip(aid_list_, oriented_aid_list_))

        all_aid_list += aid_list_
        all_aid_list += oriented_aid_list_

    config2_ = {
        'resize_dim': 'wh',
        'dim_size': (INPUT_SIZE, INPUT_SIZE),
    }
    # Pre-compute in parallel quickly so they are cached
    ibs.get_annot_chip_fpath(all_aid_list, ensure=True, config2_=config2_)

    key_list = list(result_dict.keys())
    slots = (
        len(key_list),
        num_examples,
    )

    figsize = (
        8 * slots[1],
        5 * slots[0],
    )
    fig_ = plt.figure(figsize=figsize, dpi=200)  # NOQA
    plt.grid(None)

    index = 1
    key_list = sorted(key_list)
    for row, key in enumerate(key_list):
        value_list = result_dict[key]
        for col, value in enumerate(value_list):
            aid, aid_ = value

            axes_ = plt.subplot(slots[0], slots[1], index)
            axes_.axis('off')

            chip = ibs.get_annot_chips(aid, config2_=config2_)
            chip = chip[:, :, ::-1]

            chip_ = ibs.get_annot_chips(aid_, config2_=config2_)
            chip_ = chip_[:, :, ::-1]

            canvas = np.hstack((chip, chip_))
            plt.imshow(canvas)
            index += 1

    args = (desired_note,)
    fig_filename = 'orientation.examples.predictions.%s.png' % args
    fig_path = os.path.abspath(os.path.expanduser(os.path.join('~', 'Desktop')))
    fig_filepath = os.path.join(fig_path, fig_filename)
    plt.savefig(fig_filepath, bbox_inches='tight')

    return fig_filepath


@register_ibs_method
def wbia_plugin_orientation_render_feasability(
    ibs, desired_species, desired_notes=None, **kwargs
):
    r"""
    Show examples of the prediction for each species

    Args:
        ibs       (IBEISController): IBEIS controller object
        aid_list  (list of int): A list of IBEIS Annotation IDs (aids)
        model_tag (string): Key to URL_DICT entry for this model

    Returns:
        theta_list

    CommandLine:
        python -m wbia_orientation._plugin --test-wbia_plugin_orientation_render_feasability

    Example0:
        >>> # ENABLE_DOCTEST
        >>> import wbia
        >>> import random
        >>> from wbia.init import sysres
        >>> import numpy as np
        >>> dbdir = sysres.ensure_testdb_orientation()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> value_list = [
        >>>     'right_whale_head',
        >>>     'turtle_hawksbill+head',
        >>>     'seadragon_weedy+head',
        >>>     'mobula_birostris',
        >>> ]
        >>> for desired_species in value_list:
        >>>     fig_filepath = ibs.wbia_plugin_orientation_render_feasability(
        >>>         desired_species=desired_species,
        >>>     )
        >>>     print(fig_filepath)
    """
    MAX_RANK = 12

    def rank(ibs, result):
        cm_dict = result['cm_dict']
        cm_key = list(cm_dict.keys())[0]
        cm = cm_dict[cm_key]

        query_name = cm['qname']
        qnid = ibs.get_name_rowids_from_text(query_name)

        annot_uuid_list = cm['dannot_uuid_list']
        annot_score_list = cm['annot_score_list']
        daid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
        dnid_list = ibs.get_annot_nids(daid_list)
        dscore_list = sorted(zip(annot_score_list, dnid_list), reverse=True)

        annot_ranks = []
        for rank, (dscore, dnid) in enumerate(dscore_list):
            if dnid == qnid:
                annot_ranks.append(rank)

        name_list = cm['unique_name_list']
        name_score_list = cm['name_score_list']
        dnid_list = ibs.get_name_rowids_from_text(name_list)
        dscore_list = sorted(zip(name_score_list, dnid_list), reverse=True)

        name_ranks = []
        for rank, (dscore, dnid) in enumerate(dscore_list):
            if dnid == qnid:
                name_ranks.append(rank)

        return annot_ranks, name_ranks

    def rank_min_avg(rank_dict, max_rank):
        min_x_list, min_y_list = [], []
        avg_x_list, avg_y_list = [], []
        for rank in range(max_rank):
            count_min, count_avg, total = 0.0, 0.0, 0.0
            for qaid in rank_dict:
                annot_ranks = rank_dict[qaid]
                if len(annot_ranks) > 0:
                    annot_min_rank = min(annot_ranks)
                    annot_avg_rank = sum(annot_ranks) / len(annot_ranks)
                    if annot_min_rank <= rank:
                        count_min += 1
                    if annot_avg_rank <= rank:
                        count_avg += 1
                total += 1
            percentage_min = count_min / total
            min_x_list.append(rank + 1)
            min_y_list.append(percentage_min)
            percentage_avg = count_avg / total
            avg_x_list.append(rank + 1)
            avg_y_list.append(percentage_avg)

        min_vals = min_x_list, min_y_list
        avg_vals = avg_x_list, avg_y_list

        return min_vals, avg_vals

    def get_marker(index, total):
        marker_list = ['o', 'X', '+', '*']
        num_markers = len(marker_list)
        if total <= 12:
            index_ = 0
        else:
            index_ = index % num_markers
        marker = marker_list[index_]
        return marker

    if desired_notes is None:
        desired_notes = [
            'source',
            'aligned',
            # 'squared',
            'random-01',
            'random-02',
            'random-03',
            'source*',
            'aligned*',
            # 'squared*',
            'random-01*',
            'random-02*',
            'random-03*',
        ]

    # Load any pre-computed ranks
    rank_dict_filepath = os.path.join(ibs.dbdir, 'ranks.%s.pkl' % (desired_species,))
    print('Using cached rank file: %r' % (rank_dict_filepath,))
    if os.path.exists(rank_dict_filepath):
        rank_dict = ut.load_cPkl(rank_dict_filepath)
    else:
        rank_dict = {}

    query_config_dict_dict = {
        'HS': {},
    }

    aid_dict = {}

    for desired_note in desired_notes:
        print('Processing %s' % (desired_note,))
        aid_list = ibs.get_valid_aids()

        note_list = ibs.get_annot_notes(aid_list)
        note_list = np.array(note_list)
        flag_list = note_list == desired_note.strip('*')
        aid_list = ut.compress(aid_list, flag_list)

        species_list = ibs.get_annot_species(aid_list)
        species_list = np.array(species_list)
        flag_list = species_list == desired_species
        aid_list = ut.compress(aid_list, flag_list)

        if desired_note.endswith('*'):

            all_aid_list = ibs.get_valid_aids()
            existing_species_list = ibs.get_annot_species(all_aid_list)
            existing_species_list = np.array(existing_species_list)
            existing_note_list = ibs.get_annot_notes(all_aid_list)
            existing_note_list = np.array(existing_note_list)
            delete_species_flag_list = existing_species_list == desired_species
            delete_note_flag_list = existing_note_list == desired_note
            delete_flag_list = delete_species_flag_list & delete_note_flag_list
            delete_aid_list = ut.compress(all_aid_list, delete_flag_list)

            config = {
                'orienter_algo': 'plugin:orientation',
                'orienter_weight_filepath': None,
            }
            result_list = ibs.depc_annot.get('orienter', aid_list, None, config=config)

            xtl_list = list(map(int, map(np.around, ut.take_column(result_list, 0))))
            ytl_list = list(map(int, map(np.around, ut.take_column(result_list, 1))))
            w_list = list(map(int, map(np.around, ut.take_column(result_list, 2))))
            h_list = list(map(int, map(np.around, ut.take_column(result_list, 3))))
            theta_list_ = ut.take_column(result_list, 4)
            bbox_list_ = list(zip(xtl_list, ytl_list, w_list, h_list))

            gid_list = ibs.get_annot_gids(aid_list)
            species_list = ibs.get_annot_species(aid_list)
            viewpoint_list = ibs.get_annot_viewpoints(aid_list)
            name_list = ibs.get_annot_names(aid_list)
            note_list = [desired_note] * len(aid_list)

            aid_list_ = ibs.add_annots(
                gid_list,
                bbox_list=bbox_list_,
                theta_list=theta_list_,
                species_list=species_list,
                viewpoint_list=viewpoint_list,
                name_list=name_list,
                notes_list=note_list,
            )

            delete_aid_list = list(set(delete_aid_list) - set(aid_list_))
            ibs.delete_annots(delete_aid_list)

            aid_list = aid_list_

        nid_list = ibs.get_annot_nids(aid_list)
        assert sum(np.array(nid_list) <= 0) == 0

        args = (
            len(aid_list),
            len(set(nid_list)),
            desired_species,
            desired_note,
        )
        print('Using %d annotations of %d names for species %r (note = %r)' % args)
        print('\t Species    : %r' % (set(ibs.get_annot_species(aid_list)),))
        print('\t Viewpoints : %r' % (set(ibs.get_annot_viewpoints(aid_list)),))

        if len(aid_list) == 0:
            print('\tSKIPPING')
            continue

        for qindex, qaid in tqdm.tqdm(list(enumerate(aid_list))):
            n = 1 if qindex <= 20 else 0

            qaid_list = [qaid]
            daid_list = aid_list

            print('Processing AID %d' % (qaid,))
            for query_config_label in query_config_dict_dict:
                query_config_dict = query_config_dict_dict[query_config_label]

                # label = query_config_label
                label = '%s %s' % (
                    query_config_label,
                    desired_note,
                )

                if label not in aid_dict:
                    aid_dict[label] = []
                aid_dict[label].append(qaid)

                if label not in rank_dict:
                    rank_dict[label] = {
                        'annots': {},
                        'names': {},
                    }

                flag1 = qaid not in rank_dict[label]['annots']
                flag2 = qaid not in rank_dict[label]['names']
                if flag1 or flag2:
                    query_result = ibs.query_chips_graph(
                        qaid_list=qaid_list,
                        daid_list=daid_list,
                        query_config_dict=query_config_dict,
                        echo_query_params=False,
                        cache_images=True,
                        n=n,
                    )
                    annot_ranks, name_ranks = rank(ibs, query_result)
                    rank_dict[label]['annots'][qaid] = annot_ranks
                    rank_dict[label]['names'][qaid] = name_ranks

            # if qindex % 10 == 0:
            #     ut.save_cPkl(rank_dict_filepath, rank_dict)

        ut.save_cPkl(rank_dict_filepath, rank_dict)

    #####

    rank_dict_ = {}
    for label in rank_dict:
        qaid_list = aid_dict.get(label, None)
        if qaid_list is None:
            continue

        annot_ranks = rank_dict[label]['annots']
        name_ranks = rank_dict[label]['names']

        annot_ranks_ = {}
        for qaid in annot_ranks:
            if qaid in qaid_list:
                annot_ranks_[qaid] = annot_ranks[qaid]

        name_ranks_ = {}
        for qaid in name_ranks:
            if qaid in qaid_list:
                name_ranks_[qaid] = name_ranks[qaid]

        rank_dict_[label] = {
            'annots': annot_ranks_,
            'names': name_ranks_,
        }

    fig_ = plt.figure(figsize=(20, 10), dpi=300)  # NOQA

    AVERAGE_RANDOM = True

    rank_label_list = list(rank_dict_.keys())

    rank_label_mapping = {}
    if AVERAGE_RANDOM:
        rank_label_list_ = []
        for rank_label in rank_label_list:
            if 'random' in rank_label:
                rank_label_ = rank_label.split('-')
                rank_label_ = rank_label_[:-1]
                rank_label_ = '-'.join(rank_label_)
                if rank_label.endswith('*'):
                    rank_label_ = '%s*' % (rank_label_,)

                rank_label_mapping[rank_label] = rank_label_
                if rank_label_ not in rank_label_list_:
                    rank_label_list_.append(rank_label_)
            else:
                rank_label_list_.append(rank_label)
        rank_label_list = rank_label_list_

    source_list, original_list, matched_list, unmatched_list = [], [], [], []
    label_list = []
    for desired_note in desired_notes:
        for query_config_label in query_config_dict_dict:
            label = '%s %s' % (
                query_config_label,
                desired_note,
            )

            label = rank_label_mapping.get(label, label)

            if label not in rank_label_list:
                continue

            if label in label_list:
                continue

            if desired_note == 'source':
                source_list.append(label)
            elif desired_note.endswith('*'):
                label_ = label.strip('*')
                if label_ in rank_label_list:
                    matched_list.append(label)
                else:
                    unmatched_list.append(label)
            else:
                original_list.append(label)
            label_list.append(label)

    assert len(source_list) <= 1
    color_label_list = original_list + unmatched_list
    color_list = pt.distinct_colors(len(color_label_list), randomize=False)

    color_dict = {}
    line_dict = {}

    for label in source_list:
        color_dict[label] = (0.0, 0.0, 0.0)
        line_dict[label] = '-'

    for label, color in zip(color_label_list, color_list):
        color_dict[label] = color
        if label in unmatched_list:
            line_dict[label] = '--'
        else:
            line_dict[label] = '-'

    for label in matched_list:
        label_ = label.strip('*')
        color = color_dict.get(label_, None)
        assert color is not None
        color_dict[label] = color
        line_dict[label] = '--'

    color_list = ut.take(color_dict, label_list)
    line_list = ut.take(line_dict, label_list)
    assert None not in color_list and None not in line_list

    rank_value_dict = {}
    rank_label_list = list(rank_dict_.keys())
    for label in rank_label_list:
        label_ = rank_label_mapping.get(label, label)
        if label_ not in rank_value_dict:
            rank_value_dict[label_] = {
                'a': {
                    'x': [],
                    'y': [],
                },
                'b': {
                    'x': [],
                    'y': [],
                },
            }

        # Plot 1
        annot_ranks = rank_dict_[label]['annots']
        min_vals, avg_vals = rank_min_avg(annot_ranks, MAX_RANK)
        x_list, y_list = min_vals
        # x_list, y_list = avg_vals
        rank_value_dict[label_]['a']['x'].append(x_list)
        rank_value_dict[label_]['a']['y'].append(y_list)

        # Plot 2
        name_ranks = rank_dict_[label]['names']
        min_vals, avg_vals = rank_min_avg(name_ranks, MAX_RANK)
        x_list, y_list = min_vals
        # x_list, y_list = avg_vals
        rank_value_dict[label_]['b']['x'].append(x_list)
        rank_value_dict[label_]['b']['y'].append(y_list)

    for label in rank_value_dict:
        rank_value_dict[label]['a']['x'] = np.mean(
            rank_value_dict[label]['a']['x'], axis=0
        )
        rank_value_dict[label]['a']['y'] = np.mean(
            rank_value_dict[label]['a']['y'], axis=0
        )
        rank_value_dict[label]['b']['x'] = np.mean(
            rank_value_dict[label]['b']['x'], axis=0
        )
        rank_value_dict[label]['b']['y'] = np.mean(
            rank_value_dict[label]['b']['y'], axis=0
        )

    values_list = []
    for label in label_list:
        x_list = rank_value_dict[label]['a']['x']
        y_list = rank_value_dict[label]['a']['y']
        values = (
            label,
            x_list,
            y_list,
        )
        values_list.append(values)

    axes_ = plt.subplot(121)
    axes_.set_autoscalex_on(False)
    axes_.set_autoscaley_on(False)
    axes_.set_ylabel('Percentage')
    axes_.set_xlabel('Rank')
    axes_.set_xlim([1.0, MAX_RANK])
    axes_.set_ylim([0.0, 1.0])
    zipped = list(zip(color_list, line_list, values_list))
    total = len(zipped)
    for index, (color, linestyle, values) in enumerate(zipped):
        label, x_list, y_list = values
        marker = get_marker(index, total)
        plt.plot(
            x_list,
            y_list,
            color=color,
            marker=marker,
            label=label,
            linestyle=linestyle,
            alpha=1.0,
        )

    plt.title('One-to-Many Annotations - Cumulative Match Rank')
    plt.legend(
        bbox_to_anchor=(0.0, 1.04, 1.0, 0.102),
        loc=3,
        ncol=2,
        mode='expand',
        borderaxespad=0.0,
    )

    values_list = []
    for label in label_list:
        x_list = rank_value_dict[label]['b']['x']
        y_list = rank_value_dict[label]['b']['y']
        values = (
            label,
            x_list,
            y_list,
        )
        values_list.append(values)

    axes_ = plt.subplot(122)
    axes_.set_autoscalex_on(False)
    axes_.set_autoscaley_on(False)
    axes_.set_ylabel('Percentage')
    axes_.set_xlabel('Rank')
    axes_.set_xlim([1.0, MAX_RANK])
    axes_.set_ylim([0.0, 1.0])
    zipped = list(zip(color_list, line_list, values_list))
    total = len(zipped)
    for index, (color, linestyle, values) in enumerate(zipped):
        label, x_list, y_list = values
        marker = get_marker(index, total)
        plt.plot(
            x_list,
            y_list,
            color=color,
            marker=marker,
            label=label,
            linestyle=linestyle,
            alpha=1.0,
        )

    plt.title('One-to-Many Names - Cumulative Match Rank')
    plt.legend(
        bbox_to_anchor=(0.0, 1.04, 1.0, 0.102),
        loc=3,
        ncol=2,
        mode='expand',
        borderaxespad=0.0,
    )

    label_list_ = [val.lower().replace(' ', '_') for val in label_list]
    note_str = '_'.join(label_list_)
    args = (
        desired_species,
        note_str,
    )
    fig_filename = 'orientation.matching.hotspotter.%s.%s.png' % args
    fig_path = os.path.abspath(os.path.expanduser(os.path.join('~', 'Desktop')))
    fig_filepath = os.path.join(fig_path, fig_filename)
    plt.savefig(fig_filepath, bbox_inches='tight')

    return fig_filepath


@register_ibs_method
def train_model(ibs, tag, species_list, species_mapping={}, viewpoint_mapping={}):
    import pathlib
    from os.path import join

    src_path = ibs.export_to_coco(
        species_list,
        species_mapping=species_mapping,
        viewpoint_mapping=viewpoint_mapping,
        include_parts=False,
        require_image_reviewed=True,
        include_reviews=False,
    )

    data_path = join(pathlib.Path(__file__).parent.absolute(), 'data')
    dst_path = join(data_path, 'orientation.%s.coco' % (tag,))

    ut.ensure_dir(data_path)
    ut.copy(src_path, dst_path)

    config_str = """DATASET:
      NAME: 'beluga'

    TEST:
      MODEL_FILE: 'wbia_orientation/output/%s/best.pth'
      BS: 32

    VERSION: v0""" % (
        tag,
    )

    config_filepath = 'config/orientation.%s.yaml' % (tag,)

    with open(config_filepath, 'r') as config_file:
        config_file.write(config_str)

    # python wbia_orientation/train.py --cfg wbia_orientation/config/orientation.beluga.yaml


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia_orientation._plugin --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
