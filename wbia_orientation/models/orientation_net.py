# -*- coding: utf-8 -*-
# Written by Olga Moskvyak (olga.moskvyak@hdr.qut.edu.au)

import logging
import torch.nn as nn
from torchvision import models as torchmodels
from efficientnet_pytorch import EfficientNet
import models

logger = logging.getLogger(__name__)


class OrientationNet(nn.Module):
    """CNN that normalizes orientation of the object in the image.
    Model outputs 5 floats: xc, yc, xt, yt, w
    Input:
        core_name (string): name of core model, class from torchvision.models
    """

    def __init__(self, cfg, is_train):
        super(OrientationNet, self).__init__()
        core_name = cfg.MODEL.CORE_NAME

        output_num = 5
        # Use Sigmoid activation function (output from 0 to 1) for coords
        self.sigmoid = nn.Sigmoid()

        # Load core model
        if 'hrnet' in core_name.lower():
            self.model = models.cls_hrnet.get_cls_net(cfg)
        elif 'efficientnet' in core_name.lower():
            self.model = EfficientNet.from_pretrained(core_name)
        else:
            # Load pretrained model if training
            self.model = eval('torchmodels.' + core_name)(pretrained=is_train)

        # Modify the last layer
        if 'resnet' in core_name.lower():
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, output_num)

        elif 'efficientnet' in core_name.lower():
            num_ftrs = self.model._fc.in_features
            self.model._fc = nn.Linear(num_ftrs, output_num)

        elif 'densenet' in core_name.lower():
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_ftrs, output_num)

        elif 'hrnet' in core_name.lower():
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_ftrs, output_num)

        else:
            logger.error('=> invalid core_name {}'.format(core_name))
            raise ValueError('Invalid core_name: {}'.format(core_name))

    def forward(self, x):
        out = self.model(x)
        out = self.sigmoid(out)
        return out
