# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# Written by Olga Moskvyak (olga.moskvyak@hdr.qut.edu.au)
# ------------------------------------------------------------------------------
import logging
import torch.nn as nn
from torchvision import models as torchmodels

import models

logger = logging.getLogger(__name__)


class OrientationNet(nn.Module):
    """ CNN that normalizes orientation of the object in the image
    Input:
        core_name (string): name of core feature extractor, class from torchvision.models
        predict_angle (bool, default False):
            if False then output is 5 floats: xc, yc, xt, yt, x
            if True then output is cos(theta), angle of rotation in 
                clockwise direction of the image from vertical orientation
    """
    def __init__(self, cfg, is_train):
        super(OrientationNet, self).__init__()
        self.predict_angle = cfg.MODEL.PREDICT_THETA
        core_name = cfg.MODEL.CORE_NAME
        
        if self.predict_angle:
            output_num = 1
        else:
            output_num = 5
        
        if 'hrnet' in core_name.lower():
            self.model = models.hrnet.get_pose_net(cfg, is_train)
        else:
            #if training, load pretrained model
            self.model = eval('torchmodels.'+core_name)(pretrained=is_train, progress=True)
        
        if 'resnet' in core_name.lower():
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, output_num)
            
        elif 'densenet' in core_name.lower():
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_ftrs, output_num)
        else:
            logger.error('=> invalid core_name {}'.format(core_name))
            raise ValueError('Invalid core_name: {}'.format(core_name)) 
            
        #Additional layers
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
        
    def forward(self, x): 
        out = self.model(x)
        
        if self.predict_angle:
            #For angle use Tahn activation function (output from -1 to 1)
            out = self.tanh(out)
        else:
            #For coords use Sigmoid activation function (output from 0 to 1)
            out = self.sigmoid(out)
                
        return out

