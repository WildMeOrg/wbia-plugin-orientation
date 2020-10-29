# -*- coding: utf-8 -*-
# Written by Olga Moskvyak (olga.moskvyak@hdr.qut.edu.au)

from skimage import transform
import numpy as np
import math


def midpoint(point_1, point_2):
    """Middle point between two points"""
    x1, y1 = point_1
    x2, y2 = point_2
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def add_dist_along_vector(p0, p1, dist):
    """Add distance along vector"""
    p0 = np.array(p0)
    p1 = np.array(p1)
    u = (p1 - p0) / np.linalg.norm((p1 - p0))
    return p0 + dist * u


def add_dict_perpendicular_vector(p0, p1, dist):
    """Get point in a distance perpendicular to vector"""
    p0 = np.array(p0)
    p1 = np.array(p1)
    u = (p1 - p0) / np.linalg.norm((p1 - p0))
    v1 = np.array([-u[1], u[0]])
    v2 = np.array([u[1], -u[0]])
    return p1 + dist * v1, p1 + dist * v2


def rotate_point_by_angle(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def get_object_aligned_box(xc, yc, xt, yt, w):
    """Get object aligned bounding box given
    the center point (xc, yc), the side point (xt, yt) and half width w
    Input:
        xc, yc, xt, yt, w: float or int, coordinates
    Returns:
        corner_1, corner_2, corner_3, corner_4: four tuples - coordinates
        of four corners of bounding box (not in sequantial order)
    """
    corner_1, corner_2 = add_dict_perpendicular_vector([xc, yc], [xt, yt], w)
    dist = np.linalg.norm([xc - xt, yc - yt])
    xt_, yt_ = add_dist_along_vector([xt, yt], [xc, yc], 2 * dist)
    corner_3, corner_4 = add_dict_perpendicular_vector([xc, yc], [xt_, yt_], w)

    return corner_1, corner_2, corner_3, corner_4


def plot_image_coordinates(ax, image, xc, yc, xt, yt, w):
    predicted_oa_box = get_object_aligned_box(xc, yc, xt, yt, w)
    predicted_oa_box = np.array(predicted_oa_box)

    ax.imshow(image)
    ax.plot(xc, yc, 'ro')
    ax.plot(xt, yt, 'yo')
    ax.plot(predicted_oa_box[:, 0], predicted_oa_box[:, 1], 'go')
    ax.plot(predicted_oa_box[:, 0], predicted_oa_box[:, 1], 'go-', linewidth=2)
    ax.plot(
        [predicted_oa_box[0, 0], predicted_oa_box[-1, 0]],
        [predicted_oa_box[0, 1], predicted_oa_box[-1, 1]],
        'go-',
        linewidth=2,
    )


def increase_bbox(bbox, scale, image_size, type='xyhw'):
    """Increase the size of the bounding box
    Input:
        bbox_xywh:
        scale:
        image_size: tuple of int, (h, w)
        type (string): notation of bbox: 'xyhw' or 'xyx2y2'
    """
    if type == 'xyhw':
        x1, y1, bbox_w, bbox_h = bbox
        x2 = x1 + bbox_w
        y2 = y1 + bbox_h
    else:
        x1, y1, x2, y2 = bbox
        bbox_h = y2 - y1
        bbox_w = x2 - x1
    h, w = image_size

    increase_w_by = (bbox_w * scale - bbox_w) // 2
    increase_h_by = (bbox_h * scale - bbox_h) // 2

    new_x1 = int(max(0, x1 - increase_w_by))
    new_x2 = int(min(w - 1, x2 + increase_w_by))

    new_y1 = int(max(0, y1 - increase_h_by))
    new_y2 = int(min(h - 1, y2 + increase_h_by))

    if type == 'xyhw':
        return (new_x1, new_y1, new_x2 - new_x1, new_y2 - new_y1)
    else:
        return (new_x1, new_y1, new_x2, new_y2)


def to_origin(bbox_xywh, new_origin):
    """Update coordinates of bounding box after moving the origin
    Height and width do not change.
    Coordinates are allowed to be negative and go outside of image boundary.
    Input:
        coords (list of floats): coords of bbox, [x1, y1, w, h]

    """
    bbox_xywh[0] -= new_origin[0]
    bbox_xywh[1] -= new_origin[1]
    return bbox_xywh


def rotate_coordinates(coords, angle, rotation_centre, imsize, resize=False):
    """Rotate coordinates in the image"""
    rot_centre = np.asanyarray(rotation_centre)
    angle = math.radians(angle)
    rot_matrix = np.array(
        [
            [math.cos(angle), math.sin(angle), 0],
            [-math.sin(angle), math.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    coords = transform.matrix_transform(coords - rot_centre, rot_matrix) + rot_centre

    if resize:
        rows, cols = imsize[0], imsize[1]
        corners = np.array(
            [[0, 0], [0, rows - 1], [cols - 1, rows - 1], [cols - 1, 0]], dtype=np.float32
        )
        if rotation_centre is not None:
            corners = (
                transform.matrix_transform(corners - rot_centre, rot_matrix) + rot_centre
            )

        x_shift = min(corners[:, 0])
        y_shift = min(corners[:, 1])
        coords -= np.array([x_shift, y_shift])

    return coords


def resize_coords(coords, original_size, target_size):
    """Resize coordinates
    Input:
        coords (list or tuple of floats): (x, y) coordinates
        original_size: size of image (h, w)
        target_size: target size of image (h, w)
    """
    assert isinstance(coords, (list, tuple))
    assert len(coords) % 2 == 0
    assert len(original_size) == 2
    assert len(target_size) == 2

    if type(coords) == tuple:
        coords = list(coords)

    for i in range(0, len(coords), 2):
        coords[i] = int((coords[i] / original_size[1]) * target_size[1])
        coords[i + 1] = int((coords[i + 1] / original_size[0]) * target_size[0])
    return coords


def resize_sample(sample, original_size, target_size):
    image, xc, yc, xt, yt, w, theta = sample

    # Compute second end of segment w (first end of w is in xt, yt)
    (xw_end, yw_end), _ = add_dict_perpendicular_vector([xc, yc], [xt, yt], w)

    image = transform.resize(image, target_size, order=3, anti_aliasing=True)

    # Update coordinates
    xc, yc = resize_coords((xc, yc), original_size, target_size)
    xt, yt = resize_coords((xt, yt), original_size, target_size)
    xw_end, yw_end = resize_coords((xw_end, yw_end), original_size, target_size)

    # Recompute w
    w = np.linalg.norm([xw_end - xt, yw_end - yt])

    # Recompute theta
    theta = np.arctan2(yt - yc, xt - xc) + math.radians(90)

    return image, xc, yc, xt, yt, w, theta
