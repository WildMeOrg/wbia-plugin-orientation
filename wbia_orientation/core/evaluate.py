# -*- coding: utf-8 -*-
# Written by Olga Moskvyak (olga.moskvyak@hdr.qut.edu.au)

import math
import numpy as np
import torch


def compute_theta(coords):
    """
    Compute angle of orientation theta based on output coordinates
    Input:
        coords: numpy array of shape (bs, 5), each row is [xc, yc, xt, yt, w]
    Returns:
        theta_pred: numpy array of angle theta in radians
    """
    theta_pred = np.arctan2(coords[:, 3] - coords[:, 1], coords[:, 2] - coords[:, 0])
    # Add 90 degrees to align arctan2 notation with annotations
    theta_pred += math.radians(90)
    return theta_pred


def evaluate_orientaion_coords(
    pred_coords, target_coords, target_theta, theta_thr=10, theta_source='annot'
):
    """
    Evaluate errors and accuracy of orientation using predicted coordinates.
    Predicted theta is correct if error is below threshold.
    Input:
        pred_coords: numpy array or tensor of shape (bs, 5)
                predicted values [xc, yc, xt, yt, w]
        target_coords: numpy array or tensor of shape (bs, 5)
                ground truth values [xc, yc, xt, yt, w]
        target_theta: numpy array or tensor of shape (bs),
                ground truth for angle of orientation theta in radians
        theta_thr: int, threshold in degrees when theta detection
                is considered correct
        theta_source (string): 'annot' or 'calc', source of theta,
            'annot' - get theta from gt theta annotations
            'calc' - calculate from gt coordinates.
    Returns:
        eval_dict (dictionary):
            'err_theta': mean error in degrees for angle theta
            'acc_theta': accuracy of theta prediction
            'err_xcyc': mean error in pixels for (xc, yc)
            'err_xtyt': mean error in pixels for (xt, yt)
            'err_w': mean distance in pixels between gt and predictions for w
    """
    # Convert to numpy arrays if tensors
    if type(pred_coords) == torch.Tensor:
        pred_coords = pred_coords.numpy()
    if type(target_coords) == torch.Tensor:
        target_coords = target_coords.numpy()
    if type(target_theta) == torch.Tensor:
        target_theta = target_theta.numpy()

    # Compute predicted theta (np.arctan2(yt-yc, xt-xc))
    theta_pred = np.arctan2(
        pred_coords[:, 3] - pred_coords[:, 1], pred_coords[:, 2] - pred_coords[:, 0]
    )
    if theta_source == 'annot':
        # Arctan compute angle between provided vector and vector (1, 0).
        # To align arctan output with annotations,
        # we add 90 degrees to arctan output
        theta_pred += math.radians(90)

    # True theta is computed either from ground truth (xc, yc) and (xt, yt)
    # or taken directly from annotations
    if theta_source == 'annot':
        theta_gt = target_theta
    else:
        theta_gt = np.arctan2(
            target_coords[:, 3] - target_coords[:, 1],
            target_coords[:, 2] - target_coords[:, 0],
        )

    # Comvert to degrees
    theta_pred = np.rad2deg(theta_pred)
    theta_gt = np.rad2deg(theta_gt)

    # Normalize angles to range between -180 and 180 degrees
    np_norm_theta = np.vectorize(normalize_theta)
    err_theta = np.abs(np_norm_theta(theta_pred) - np_norm_theta(theta_gt))
    correct_idx = err_theta <= theta_thr
    acc_theta = correct_idx.sum() / len(theta_gt)

    err_xcyc = np.linalg.norm(target_coords[:, :2] - pred_coords[:, :2])
    err_xtyt = np.linalg.norm(target_coords[:, 2:4] - pred_coords[:, 2:4])
    err_w = np.abs(target_coords[:, 4] - pred_coords[:, 4])

    eval_dict = {
        'err_theta': np.mean(err_theta),
        'acc_theta': acc_theta,
        'err_xcyc': np.mean(err_xcyc),
        'err_xtyt': np.mean(err_xtyt),
        'err_w': np.mean(err_w),
        'err_idx': ~correct_idx,
    }
    return eval_dict


def normalize_theta(theta, degrees=True):
    """Normalize angle theta to -180 and 180 degrees
    Input:
        theta (float): angle in degrees or radians
        degrees (bool, default True): if True theta in degrees else in radians
    """
    if degrees:
        pi = 180.0
    else:
        pi = math.pi

    if theta >= 2 * pi or theta <= -2 * pi:
        theta = theta % (2 * pi)

    if theta <= pi and theta > -pi:
        return theta

    elif theta > pi:
        theta -= 2 * pi
        return theta

    else:
        theta += 2 * pi
        return theta
