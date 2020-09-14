# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# Written by Olga Moskvyak (olga.moskvyak@hdr.qut.edu.au)
# ------------------------------------------------------------------------------

import numpy as np

import math

def midpoint(point_1, point_2):
    """Middle point between two points
    """
    x1, y1 = point_1
    x2, y2 = point_2
    return ((x1 + x2)/2, (y1 + y2)/2)

def add_dist_along_vector(p0, p1, dist):
    """Add distance along vector
    """
    p0 = np.array(p0)
    p1 = np.array(p1)
    u = (p1 - p0) / np.linalg.norm((p1-p0))
    return p0 + dist * u

def add_dict_perpendicular_vector(p0, p1, dist):
    """Get point in a distance perpendicular to vector
    """
    p0 = np.array(p0)
    p1 = np.array(p1)
    u = (p1 - p0) / np.linalg.norm((p1-p0))
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
    """Get object aligned bounding box given center point (xc, yc), side point (xt, yt) and half width w
    Input:
        xc, yc, xt, yt, w: float or int, coordinates
    Returns:
        corner_1, corner_2, corner_3, corner_4: four tuples - coordinates of four corners of bounding box (not in sequantial order)
    """
    corner_1, corner_2 = add_dict_perpendicular_vector([xc, yc], [xt, yt], w)
    dist = np.linalg.norm([xc-xt, yc-yt])
    xt_, yt_ = add_dist_along_vector([xt, yt], [xc, yc], 2*dist)
    corner_3, corner_4 = add_dict_perpendicular_vector([xc, yc], [xt_, yt_], w)
    
    return corner_1, corner_2, corner_3, corner_4

def plot_image_coordinates(ax, image, xc, yc, xt, yt, w, theta):
    predicted_oa_box = get_object_aligned_box(xc, yc, xt, yt, w)
    predicted_oa_box = np.array(predicted_oa_box)

    ax.imshow(image)
    ax.plot(xc, yc, 'r*')
    ax.plot(xt, yt, 'y*')
    ax.plot(predicted_oa_box[:,0], predicted_oa_box[:,1], 'go')
    
def increase_bbox(bbox_xyx2y2, scale, image_size):
    """Increase the size of the bounding box
    Input:
        bbox_xyx2y2:
        scale:
        image_size: tuple of int, (h, w)
    """
    x1, y1, x2, y2 = bbox_xyx2y2
    h, w = image_size
    bbox_h = y2 - y1
    bbox_w = x2 - x1
    
    increase_w_by = (bbox_w * scale - bbox_w) // 2
    increase_h_by = (bbox_h * scale - bbox_h) // 2
    
    new_x1 = int(max(0, x1 - increase_w_by))
    new_x2 = int(min(w-1, x2 + increase_w_by))
    
    new_y1 = int(max(0, y1 - increase_h_by))
    new_y2 = int(min(h-1, y2 + increase_h_by))
    
    return (new_x1, new_y1, new_x2, new_y2) 