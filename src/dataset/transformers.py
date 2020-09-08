#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 11:51:51 2019

@author: olga
"""
import torch
from skimage import transform
import random
import math
import numpy as np
from torchvision.transforms import functional as F

class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
        left_right_pairs: list of pairs for left-right annotations, e.g. [(3, 5), (4, 6)]
    """

    def __init__(self, p=0.5, left_right_pairs=[]):
        self.p = p
        self.left_right_pairs = left_right_pairs

    def __call__(self, sample):
        """
        Args:
            img : numpy array, Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        image, coords, vis = sample
        
        if random.random() < self.p:
            #Flip image
            image = np.fliplr(image) # F.hflip(image)
            #Flip simmetrical coordinates
            for c, coord in enumerate(coords):
                if vis[c][0] > 0:
                    coords[c][:2] = [int(image.shape[1]-coord[0]), coord[1]]
                    
            #change left and right annotations for symmetrical parts
            if len(self.left_right_pairs) > 0:
                for (left_idx, right_idx) in self.left_right_pairs:
                    temp_vis, temp_coord = vis[right_idx], coords[right_idx]
                    vis[right_idx], coords[right_idx] = vis[left_idx], coords[left_idx]
                    vis[left_idx], coords[left_idx] = temp_vis, temp_coord
                
            #print('after coords and vis', coords, vis)
        
            return image, coords, vis
        return sample

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
        image, coords, vis = sample

        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        
        if h > new_h and w > new_w:
            top = int(round((h - new_h) / 2.))
            left = int(round((w - new_w) / 2.))
    
            image = image[top: top + new_h, left: left + new_w]
            
            coords = coords - np.array([left, top, 0])
            for c, coord in enumerate(coords):
                if coord[0] < 0 or coord[0] >= new_w:
                    vis[c] = 0.
                if coord[1] < 0 or coord[1] >= new_h:
                    vis[c] = 0.

        return image, coords, vis
   

class Rotate(object):
    """Rotate the image in a sample and a corresponding heatmap by a random angle.
    TODO: rotate coordinates

    Args:
        max_angle (int or float): Maximum rotation angle in degrees
    
    Returned images have dtype float in range (0,1) because of applied transform
    """
    def __init__(self, max_angle):
        assert isinstance(max_angle, (int, float))
        self.max_angle = max_angle

    def __call__(self, sample):
        image, coords, vis = sample
        angle = random.randint(-self.max_angle, self.max_angle)
        #print(angle)
#        print(type(image), angle)
        image = transform.rotate(image, angle, mode='edge')
        
        #Define transformations
        rot_matrix = np.array([[ math.cos(math.radians(angle)), math.sin(math.radians(angle)), 0],
                              [-math.sin(math.radians(angle)), math.cos(math.radians(angle)), 0],
                              [0,0,1]])

        #Apply transformations to coordinates
        rotation_centre = (image.shape[1] // 2, image.shape[0] // 2) 
        coords[:,:2] = transform.matrix_transform(coords[:,:2] - rotation_centre, rot_matrix) + rotation_centre
        
        for c, coord in enumerate(coords):
            if coord[0] < 0 or coord[0] >= image.shape[1]:
                vis[c] = 0.
            if coord[1] < 0 or coord[1] >= image.shape[0]:
                vis[c] = 0.

        return image, coords, vis
      
        
class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Source: https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html#ToTensor

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.
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
        indices (list of integers): indices of sample to normalize
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
    
