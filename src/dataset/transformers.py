# ------------------------------------------------------------------------------
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
            theta += math.radians(180)
        
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
    
def rotate_coordinates(coords, angle, rotation_centre, imsize, resize=False):
    """Rotate coordinates in the image
    """
    rotation_centre = np.asanyarray(rotation_centre)
    rot_matrix = np.array([[  math.cos(math.radians(angle)), math.sin(math.radians(angle)), 0],
                           [ -math.sin(math.radians(angle)), math.cos(math.radians(angle)), 0],
                           [  0,0,1]])
    coords = transform.matrix_transform(coords - rotation_centre, rot_matrix) + rotation_centre
                
    if resize:
        rows, cols = imsize[0], imsize[1]
        corners = np.array([[0, 0], [0, rows - 1], [cols - 1, rows - 1], [cols - 1, 0]], dtype=np.float32)
        if rotation_centre is not None:
            corners = transform.matrix_transform(corners - rotation_centre, rot_matrix) + rotation_centre
    
        x_shift = min(corners[:,0])
        y_shift = min(corners[:,1])       
        coords -= np.array([x_shift, y_shift])
        
    return coords  

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, coords, vis = sample

        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        
        if h > new_h and w > new_w:
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
    
            image = image[top: top + new_h, left: left + new_w]
            
            coords = coords - np.array([left, top, 0])
            for c, coord in enumerate(coords):
                if coord[0] < 0 or coord[0] >= new_w:
                    vis[c] = 0.
                if coord[1] < 0 or coord[1] >= new_h:
                    vis[c] = 0.

        return image, coords, vis


class Resize(object):
    """Resize only square image into square image of different size
    Reason: cannot determine how width changes if resizing a rectangular image
    """
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size
        
    def __call__(self, sample):
        image, xc, yc, xt, yt, w, theta = sample
        
        assert image.shape[0] == image.shape[1]
        original_size = image.shape[0]
        image = transform.resize(image, 
                                 (self.output_size, self.output_size), 
                                 order=3, 
                                 anti_aliasing=True)
        
        #Update coordinates
        xc = int((xc / original_size) * self.output_size)
        yc = int((yc / original_size) * self.output_size)
        xt = int((xt / original_size) * self.output_size)
        yt = int((yt / original_size) * self.output_size)
        w  = int((w / original_size) * self.output_size)
        
        return image, xc, yc, xt, yt, w, theta

class CropObjectArea(object):
    def __init__(self, noise):
        self.noise = noise
     
    def __call__(self, sample):
        image, xc, yc, xt, yt, w, theta = sample
        
        #Get object-aligned bounding box
        corners = get_object_aligned_box(xc, yc, xt, yt, w)
        
        #Get bounding box around object-aligned box
        corners = np.asarray(corners)
        print('corners shape', corners.shape)
        x_min = max(0, min(corners[:,0]))
        y_min = max(0, min(corners[:,1]))
        x_max = min(max(corners[:,0]), image.shape[1])
        y_max = min(max(corners[:,1]), image.shape[0])
                
        #Make it square (add black border if does not fit)
        #Manipulate coordinates of bounding box
        #Add noise to randomize
        
        #TODO make coordinates int
        #Crop image and coordinates
        image = image[y_min:y_max, x_min:x_max]
        xc -= x_min
        yc -= y_min
        xt -= x_min
        yt -= y_min
        #Width and theta do not change
        
        return image, xc, yc, xt, yt, w, theta
        
        
    
class CenterCrop(object):
    """Crop in the center the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, xc, yc, xt, yt, w, theta = sample

        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        
        if h > new_h and w > new_w:
            top = int(round((h - new_h) / 2.))
            left = int(round((w - new_w) / 2.))
    
            image = image[top: top + new_h, left: left + new_w]
            

        return image, xc, yc, xt, yt, w, theta
         
        
class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Source: https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html#ToTensor

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8
    In the other cases, tensors are returned without scaling.
    
    Input:
        indices (list of integers): indices of images to convert in sample
    """
    def __init__(self, indices=[0]):
        self.indices = indices
        
    def __call__(self, sample):
        for idx in self.indices:
            sample[idx] = F.to_tensor(sample[idx].copy()).type(torch.FloatTensor)
        return sample

        
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
        indices (list of integers): indices of image to normalize in sample
    """

    def __init__(self, mean, std, inplace=False, indices=[0]):
        self.mean = mean
        self.std = std
        self.inplace = inplace
        self.indices = indices

    def __call__(self, sample):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        for idx in self.indices:
            sample[idx] = F.normalize(sample[idx], self.mean, self.std, self.inplace)
        return sample
    
