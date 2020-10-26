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
import tqdm
import os

(print, rrr, profile) = ut.inject2(__name__)

_, register_ibs_method = controller_inject.make_ibs_register_decorator(__name__)

register_api = controller_inject.get_wbia_flask_api(__name__)
register_route = controller_inject.get_wbia_flask_route(__name__)

MODEL_URLS = {
    'seaturtle': 'https://wildbookiarepository.azureedge.net/models/orientation.seaturtle.h5',
    'seadragon': 'https://wildbookiarepository.azureedge.net/models/orientation.seadragib.h5',
    'whaleshark': 'https://wildbookiarepository.azureedge.net/models/orientation.whaleshark.h5',
    'mantaray': 'https://wildbookiarepository.azureedge.net/models/orientation.mantaray.h5',
    'spotteddolphin': 'https://wildbookiarepository.azureedge.net/models/orientation.spotteddolphin.h5',
    'hammerhead': 'https://wildbookiarepository.azureedge.net/models/orientation.hammerhead.h5',
    'rightwhale': 'https://wildbookiarepository.azureedge.net/models/orientation.rightwhale.h5'
}


register_preproc_image = controller_inject.register_preprocs['image']
register_preproc_annot = controller_inject.register_preprocs['annot']
# register_preproc_part  = controller_inject.register_preprocs['part']


@register_ibs_method
@register_api('/api/plugin/orientation/', methods=['GET'])
def wbia_plugin_orientation_inference(ibs, aid_list):
    r"""
    A "Hello world!" example for the WBIA identification plug-in.

    Args:
        ibs (WBIAController):  wbia controller object
        aid_list (list of int): A list of IBEIS Annotation IDs (aids)

    Returns:
        list: gid_list

    CommandLine:


    RESTful:
        Method: GET

        URL:    /api/plugin/orientation/

    Example:
        >>> # ENABLE_DOCTEST

    """
    pass


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia_id._plugin --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
