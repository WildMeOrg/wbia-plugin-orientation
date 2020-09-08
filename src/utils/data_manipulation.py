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
