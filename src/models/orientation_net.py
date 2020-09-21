# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# Written by Olga Moskvyak (olga.moskvyak@hdr.qut.edu.au)
# ------------------------------------------------------------------------------

import logging
import torch
import torch.nn as nn
from torchvision import models


logger = logging.getLogger(__name__)


class OrientationNet(nn.Module):
    """ CNN that normalizes orientation of the object in the image
    Input:
        core_name (string): name of core feature extractor, class from torchvision.models
        output_type (string): 'coords' or 'angle'
            if 'coords' then output is 5 floats: xc, yc, xt, yt, x
            if 'angle' then output is theta, angle of rotation in 
                clockwise direction of the image from vertical orientation
    """
    def __init__(self, core_name='resnet18', output_type='coords'):
        super(OrientationNet, self).__init__()
        
        assert output_type in ['coords', 'angle']
        if output_type == 'coords':
            output_num = 5
        else:
            output_num = 1
        
        self.model = eval('models.'+core_name)(pretrained=False, progress=True)
        #set_parameter_requires_grad(self.model, feature_extract)
        
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
        
        
    def forward(self, x): 
        out = self.model(x)
        out = self.sigmoid(out)
                
        return out
