# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------
# Differnce from PyTorch Transformers (https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html)
# 1. Image in a sample is numpy array (not PIL Image)
# 2. Custom transformers augment image and corresponding coordinates the same way
# ------------------------------------------------------------------------------

import torch
from skimage import transform
import random
import math
import numpy as np
from torchvision.transforms import functional as F
from torchvision.transforms import RandomAffine
from utils.data_manipulation import get_object_aligned_box
from utils.data_manipulation import increase_bbox
from utils.data_manipulation import resize_sample
from utils.data_manipulation import rotate_coordinates


class RandomHorizontalFlip(object):
    """Horizontally flip numpy image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
        left_right_pairs: list of pairs for left-right annotations, e.g. [(3, 5), (4, 6)]
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        """
        Args:
            img : numpy array, Image to be flipped.

        Returns:
            image: andomly flipped image.
        """
        image, xc, yc, xt, yt, w, theta = sample
        if random.random() < self.p:
            #Flip image
            image = np.fliplr(image) # F.hflip(image)
            
            #Flip xc and xt coordinates
            xc = int(image.shape[1]-xc)
            xt = int(image.shape[1]-xt)
            
            #Y-coordinates does not change after horizontal flip
            #W - width does not change after horizontal flip
            
            #Angle theta change sign
            theta = -theta
        
            return image, xc, yc, xt, yt, w, theta
        return sample
    
class RandomVerticalFlip(object):
    """Vertically flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
        left_right_pairs: list of pairs for left-right annotations, e.g. [(3, 5), (4, 6)]
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        """
        Args:
            img : numpy array, Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        image, xc, yc, xt, yt, w, theta = sample
        if random.random() < self.p:
            #Flip image
            image = np.flipud(image)
            
            #Flip xc and xt coordinates
            yc = int(image.shape[0]-yc)
            yt = int(image.shape[0]-yt)
            
            #X-coordinates does not change after vertical flip
            #W - width does not change after vertical flip
            
            #Angle theta change sign
            theta = math.radians(180) - theta
        
            return image, xc, yc, xt, yt, w, theta
        return sample
    
class RandomRotate(RandomAffine):
    """Rotate image and coorresponding coordinates the same way (extention of RandomAffine)
    """
    def __init__(self, degrees, mode = 'constant'):
        super().__init__(degrees=degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0)
        self.mode = mode
        
    def __call__(self, sample):
        image, xc, yc, xt, yt, w, theta = sample
        imsize_rc = image.shape[:2]
        angle, _, _, _ = self.get_params(self.degrees, 
                                        self.translate, 
                                        self.scale, 
                                        self.shear, 
                                        image.shape)
        
        #Transform image: rotate and then scale and translate
        #Use 'rotate' function as it has 'resize' parameter to resize image and avoid any cropping
        #and it has center parameter for the center of rotation
        rotation_centre = np.asarray([xc, yc])
        image = transform.rotate(image, 
                                 angle, 
                                 center = rotation_centre,
                                 mode=self.mode, 
                                 resize=True)
        
        #Apply the same transformations to coordinates
        coords = np.array([[xc, yc], [xt, yt]])
        coords = rotate_coordinates(coords, angle, rotation_centre, imsize_rc, resize=True)
        
        xc, yc, xt, yt = coords.ravel().tolist()
        
        #Theta is affected by rotation
        # Width is not affected by rotation
        theta -= math.radians(angle)
        
        return image, xc, yc, xt, yt, w, theta
    
class RandomScale(RandomAffine):
    """Scale image and coorresponding coordinates the same way (extention of RandomAffine)
    """
    def __init__(self, scale, mode = 'constant'):
        super().__init__(degrees=0, translate=None, scale=scale, shear=None, resample=False, fillcolor=0)
        self.mode = mode
        
    def __call__(self, sample):
        image, xc, yc, xt, yt, w, theta = sample
        _, _, scale, _ = self.get_params(self.degrees, 
                                        self.translate, 
                                        self.scale, 
                                        self.shear, 
                                        image.shape)
        
        image = transform.rescale(image, 
                                 scale, 
                                 mode=self.mode, 
                                 multichannel=True)
                
        #Apply the same transformations to coordinates
        coords = np.array([[xc, yc], [xt, yt]])
        coords *= scale
        
        xc, yc, xt, yt = coords.ravel().tolist()
        
        #Theta is not affected by rotation       
        #Width is only affected by scale
        w *= scale
        
        return image, xc, yc, xt, yt, w, theta

class Resize(object):
    """Resize image and corresponding coordinates
    Input:
        output_size: int, tuple or list, output shape of image
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple, list))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        
    def __call__(self, sample): 
        image, xc, yc, xt, yt, w, theta = sample        
        resized_sample = resize_sample(sample, image.shape[:2], self.output_size)       
        return resized_sample
    
    
class ResizeKeepRatio(object):
    """Resize image and corresponding coordinates
    Input:
        min_size: int, target length of the miminum side
    """
    def __init__(self, min_size):
        assert isinstance(min_size, int)
        self.min_size = min_size
        
    def __call__(self, sample):            
        image, xc, yc, xt, yt, w, theta = sample
        
        #Compute output size
        if image.shape[0] <= image.shape[1]:
            output_size = (self.min_size, int(image.shape[1] * self.min_size / image.shape[0]))
        else:
            output_size = (int(image.shape[0] * self.min_size / image.shape[1]), self.min_size)
        
        resized_sample = resize_sample(sample, image.shape[:2], output_size)        
        return resized_sample

class CropObjectAlignedArea(object):
    """Crop bounding rectange (axis-aligned) around object-aligned rectangle
     with some noise to randomise
     Input:
         noise (float from 0 to 1, default 0.): amount of noise in percentage to the w
         scale (float, default 1.): increase the size of bounding box by scale
    """
    def __init__(self, noise=0., scale=1.):
        assert isinstance(noise, float)
        assert noise >=0. and noise <=1.
        assert isinstance(scale, float)
        
        self.noise = noise
        self.scale = scale
     
    def __call__(self, sample):
        image, xc, yc, xt, yt, w, theta = sample
        
        #Get object-aligned bounding box
        corners = get_object_aligned_box(xc, yc, xt, yt, w)
        max_noise = int(self.noise * w)
        
        #Get bounding box around object-aligned box
        corners = np.asarray(corners)
        x_min = int(max(0, min(corners[:,0]) + random.randint(-max_noise, max_noise))) 
        y_min = int(max(0, min(corners[:,1]) + random.randint(-max_noise, max_noise)))
        x_max = int(min(max(corners[:,0]) + random.randint(-max_noise, max_noise), image.shape[1])) 
        y_max = int(min(max(corners[:,1]) + random.randint(-max_noise, max_noise), image.shape[0]))
                
        x_min, y_min, x_max, y_max = increase_bbox((x_min, y_min, x_max, y_max),
                                                   self.scale,
                                                   image.shape[:2],
                                                   type='xyx2y2')

        #Crop image and coordinates
        image = image[y_min:y_max, x_min:x_max]
        xc -= x_min
        yc -= y_min
        xt -= x_min
        yt -= y_min
        
        return image, xc, yc, xt, yt, w, theta
                 
        
class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Source: https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html#ToTensor

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8
    In the other cases, tensors are returned without scaling.
    
    """
        
    def __call__(self, sample):
        image, xc, yc, xt, yt, w, theta = sample
        
        image = F.to_tensor(image.copy()).type(torch.float32)
        xc = torch.tensor(xc, dtype=torch.float32)
        yc = torch.tensor(yc, dtype=torch.float32)
        xt = torch.tensor(xt, dtype=torch.float32)
        yt = torch.tensor(yt, dtype=torch.float32)
        w  = torch.tensor(w, dtype=torch.float32)
        theta = torch.tensor(theta, dtype=torch.float32)
        return image, xc, yc, xt, yt, w, theta

        
class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
        input_size (int): size of input in pixels to scale coordinates
    """

    def __init__(self, mean, std, inplace=False, input_size=256):
        self.mean = mean
        self.std = std
        self.inplace = inplace
        self.input_size = input_size

    def __call__(self, sample):
        image, xc, yc, xt, yt, w, theta = sample 
        
        image = F.normalize(image, self.mean, self.std, self.inplace)
        xc /= self.input_size
        yc /= self.input_size
        xt /= self.input_size
        yt /= self.input_size
        w /= self.input_size
        
        return image, xc, yc, xt, yt, w, theta
    
