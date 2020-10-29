# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from wbia.control import controller_inject
from wbia.constants import IMAGE_TABLE, ANNOTATION_TABLE
from wbia.constants import CONTAINERIZED, PRODUCTION  # NOQA
from wbia import dtool as dt
import numpy as np
import utool as ut
import vtool as vt
import wbia
import os
import sys
import torch

this_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(this_dir, '..', 'wbia_orientation'))

from dataset.animal_wbia import AnimalWbiaDataset
import torchvision.transforms as transforms
from config.default import _C as cfg
from train import _make_model, _model_to_gpu
from utils.data_manipulation import resize_oa_box


(print, rrr, profile) = ut.inject2(__name__)

_, register_ibs_method = controller_inject.make_ibs_register_decorator(__name__)

register_api = controller_inject.get_wbia_flask_api(__name__)
register_route = controller_inject.get_wbia_flask_route(__name__)

MODEL_URLS = {
    'turtle_hawksbill+head': 'https://wildbookiarepository.azureedge.net/models/orientation.seaturtle.h5',
    'seadragon': 'https://wildbookiarepository.azureedge.net/models/orientation.seadragib.h5',
    'whaleshark': 'https://wildbookiarepository.azureedge.net/models/orientation.whaleshark.h5',
    'mantaray': 'https://wildbookiarepository.azureedge.net/models/orientation.mantaray.h5',
    'spotteddolphin': 'https://wildbookiarepository.azureedge.net/models/orientation.spotteddolphin.h5',
    'hammerhead': 'https://wildbookiarepository.azureedge.net/models/orientation.hammerhead.h5',
    'rightwhale': 'https://wildbookiarepository.azureedge.net/models/orientation.rightwhale.h5',
}

CONFIGS = {
    'turtle_hawksbill+head': 'wbia_orientation/config/seaturtle_heads.yaml',
    'seadragon': 'experiments/seadragon.yaml',
    'whaleshark': 'experiments/whaleshark.yaml',
    'mantaray': 'wbia_orientation/config/mantaray.yaml',
    'spotteddolphin': 'experiments/spotteddolphin.yaml',
    'hammerhead': 'experiments/hammerhead.yaml',
    'rightwhale': 'experiments/rightwhale.yaml',
}

register_preproc_image = controller_inject.register_preprocs['image']
register_preproc_annot = controller_inject.register_preprocs['annot']
# register_preproc_part  = controller_inject.register_preprocs['part']


@register_ibs_method
def wbia_plugin_detect_oriented_box(ibs, aid_list, species, use_gpu=False):
    r"""
    Detect orientation of provided images

    Args:
        ibs (WBIAController):  wbia controller object
        aid_list (int): annot ids specifying the input
        species (string): type of species

    Returns:
        list: gid_list

    CommandLine:
        python -m wbia_orientation._plugin wbia_plugin_detect_oriented_box

    Example:
        >>> # ENABLE_DOCTEST
        >>> import wbia
        >>> import wbia_orientation
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_orientation()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_valid_aids()
        >>> aid_list = aid_list[:3]
        >>> wbia_orientation._plugin.wbia_plugin_detect_oriented_box(ibs, aid_list, 'mantaray')
        >>> Loaded pretrained weights for efficientnet-b4
            TODO

    """
    # A. Load config and model
    # TODO check how to define species
    cfg = _load_config(species, use_gpu)
    model = _load_model(cfg)

    # B. Preprocess image to model input
    test_loader, test_dataset, bboxes = ibs.orientation_load_data(
        aid_list, cfg.MODEL.IMSIZE, cfg
    )

    # C. Compute output
    outputs = []
    model.eval()
    with torch.no_grad():
        for images in test_loader:
            if cfg.USE_GPU:
                images = images.cuda(non_blocking=True)

            # Compute output of Orientation Network
            output = model(images.float())

            # TODO add flips as in validate function

            outputs.append(output)

    # Post-processing
    outputs = orientation_post_proc(outputs, bboxes)

    return outputs


@register_ibs_method
def orientation_load_data(ibs, aid_list, target_imsize, cfg):
    """
    Preprocess images by cropping bounding box, convert to tensor and
    normalize

    Returns:
        prep_images (tensor):

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
        drop_last=False
    )
    print('Loaded {} samples'.format(len(test_dataset)))

    return test_loader, test_dataset, bboxes


def orientation_post_proc(output, bboxes):
    """ Post processing of model output
    """
    # Concatenate and convert to numpy
    output = torch.cat(output, dim=0).numpy()

    # Resize coords back to original size
    for i in range(len(output)):
        # Each row in output is an array of 5 [xc, yc, xt, yt, w]
        # Bboxes is of format (x, y, w, h) while target size is [h, w]
        output[i] = resize_oa_box(output[i],
                                  original_size=[1., 1.],
                                  target_size=[bboxes[i][3], bboxes[i][2]])

    # Convert to list
    output = output.tolist()
    return output


def _load_config(species, use_gpu):
    config_file = CONFIGS[species]
    cfg.defrost()
    cfg.merge_from_file(config_file)
    cfg.USE_GPU = use_gpu
    cfg.freeze()
    return cfg


def _load_model(cfg):
    model = _make_model(cfg, is_train=False)

    import torch

    if cfg.USE_GPU:
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE))
    else:
        model.load_state_dict(
            torch.load(cfg.TEST.MODEL_FILE, map_location=torch.device('cpu'))
        )
    print('Loaded model from {}'.format(cfg.TEST.MODEL_FILE))
    model = _model_to_gpu(model, cfg)
    return model


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia_id._plugin --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
