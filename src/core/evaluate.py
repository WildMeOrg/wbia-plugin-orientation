# -----------------------------------------------------------------------------
# Licensed under the MIT License.
# Written by Olga Moskvyak (olga.moskvyak@hdr.qut.edu.au)
# -----------------------------------------------------------------------------
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
    theta_pred = np.arctan2(coords[:, 3] - coords[:, 1],
                            coords[:, 2] - coords[:, 0])
    # Add 90 degrees to align arctan2 notation with annotations
    theta_pred += math.radians(90)
    return theta_pred


def evaluate_orientaion_coords(output, target_coords, target_theta,
                               theta_thr=10, theta_source='annot'):
    '''
    Evaluate errors and accuracy of orientation from predicted coordinates.
    Predicted theta is correct if error is below threshold.
    Input:
        output: numpy array or tensor of shape (bs, 5)
                predicted values [xc, yc, xt, yt, w]
        target: numpy array or tensor of shape (bs, 5)
                ground truth values [xc, yc, xt, yt, w]
        target_theta: array of shape (bs),
                ground truth for angle of orientation theta in radians
        theta_thr (int): threshold in degrees
                         when theta detection is considered correct
        theta_source (string): 'annot' or 'calc', source of theta,
            get from ground truth annotations or calculate from gt coordinates.
            Arctan compute angle between provided vector and vector (1, 0).
            To align arctan output with annotations,
            we add 90 degrees to arctan output
    Returns:
        eval_dict (dictionary):
            'err_theta': mean error in degrees for angle theta
            'acc_theta': accuracy of theta prediction
            'err_xcyc': mean error in pixels for (xc, yc)
            'err_xtyt': mean error in pixels for (xt, yt)
            'err_w': mean distance in pixels between gt and predictions for w
    '''
    # Convert to numpy arrays if tensors
    if type(output) == torch.Tensor:
        output = output.numpy()
    if type(target_coords) == torch.Tensor:
        target_coords = target_coords.numpy()
    if type(target_theta) == torch.Tensor:
        target_theta = target_theta.numpy()

    # Compute predicted truth theta (np.arctan2(yt-yc, xt-xc))
    theta_pred = np.arctan2(output[:, 3] - output[:, 1],
                            output[:, 2] - output[:, 0])
    if theta_source == 'annot':
        theta_pred += math.radians(90)

    # True theta is computed either from ground truth (xc, yc) and (xt, yt)
    # or taken directly from annotations
    if theta_source == 'annot':
        theta_gt = target_theta
    else:
        theta_gt = np.arctan2(target_coords[:, 3] - target_coords[:, 1],
                              target_coords[:, 2] - target_coords[:, 0])

    # Comvert to angles
    theta_pred = np.rad2deg(theta_pred)
    theta_gt = np.rad2deg(theta_gt)

    np_norm_theta = np.vectorize(normalize_theta)
    err_theta = np.abs(np_norm_theta(theta_pred) - np_norm_theta(theta_gt))
    acc_theta = (err_theta < theta_thr).sum() / len(theta_gt)

    err_xcyc = np.linalg.norm(target_coords[:, :2] - output[:, :2])
    err_xtyt = np.linalg.norm(target_coords[:, 2:4] - output[:, 2:4])
    err_w = np.abs(target_coords[:, 4] - output[:, 4])

    eval_dict = {
            'err_theta': np.mean(err_theta),
            'acc_theta': acc_theta,
            'err_xcyc': np.mean(err_xcyc),
            'err_xtyt': np.mean(err_xtyt),
            'err_w': np.mean(err_w),
            }
    return eval_dict


def evaluate_orientaion_theta(theta_pred, theta_gt, theta_thr=10):
    '''
    Evaluate errors and accuracy of orientation detection from predicted theta
    Input:
        theta_pred: numpy array or tensor of shape (bs),
                    ground truth for cosine of theta
        theta_gt: numpy array or tensor of shape (bs),
                  ground truth for cosine of theta
        theta_thr (int): threshold for tolerated error in degrees
    Returns:
        eval_dict (dictionary):
            'err_theta': mean error in degrees for angle theta
            'acc_theta': accuracy of theta prediction,
                         considered correct if error is below threshold
    '''
    # Convert to numpy arrays if tensors
    if type(theta_pred) == torch.Tensor:
        theta_pred = theta_pred.numpy()
    if type(theta_gt) == torch.Tensor:
        theta_gt = theta_gt.numpy()

    # Comvert to angles
    theta_pred = np.rad2deg(np.arccos(theta_pred))
    theta_gt = np.rad2deg(np.arccos(theta_gt))

    err_theta = np.abs(theta_pred - theta_gt)
    acc_theta = (err_theta < theta_thr).sum() / len(theta_gt)

    eval_dict = {
            'err_theta': np.mean(err_theta),
            'acc_theta': acc_theta
            }
    return eval_dict


def normalize_theta(theta, degrees=True):
    '''Normalize angle theta to be between -180 and 180 degrees
    Input:
        theta (float): angle in degrees
        degrees (bool, default True): if True theta in degrees else in radians
    '''
    if degrees:
        pi = 180.
    else:
        pi = math.pi

    if theta >= 2*pi or theta <= -2*pi:
        theta = theta % (2*pi)

    if theta <= pi and theta > -pi:
        return theta

    elif theta > pi:
        theta -= 2*pi
        return theta

    else:
        theta += 2*pi
        return theta
