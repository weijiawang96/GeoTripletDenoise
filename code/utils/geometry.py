import math
import numpy as np
from sklearn.decomposition import PCA


def get_principle_dirs(pts):

    pts_pca = PCA(n_components=3)
    pts_pca.fit(pts)
    principle_dirs = pts_pca.components_
    principle_dirs /= np.linalg.norm(principle_dirs, 2, axis=0)

    return principle_dirs


def pca_alignment(pts, random_flag=False):

    pca_dirs = get_principle_dirs(pts)

    if random_flag:

        pca_dirs *= np.random.choice([-1, 1], 1)

    rotate_1 = compute_roatation_matrix(pca_dirs[2], [0, 0, 1], pca_dirs[1])
    pca_dirs = np.array(rotate_1 * pca_dirs.T).T
    rotate_2 = compute_roatation_matrix(pca_dirs[1], [1, 0, 0], pca_dirs[2])
    pts = np.array(rotate_2 * rotate_1 * np.matrix(pts.T)).T

    inv_rotation = np.array(np.linalg.inv(rotate_2 * rotate_1))

    return pts, inv_rotation


def compute_roatation_matrix(sour_vec, dest_vec, sour_vertical_vec=None):
    # http://immersivemath.com/forum/question/rotation-matrix-from-one-vector-to-another/
    if (
        np.linalg.norm(np.cross(sour_vec, dest_vec), 2) == 0
        or np.abs(np.dot(sour_vec, dest_vec)) >= 1.0
    ):
        if np.dot(sour_vec, dest_vec) < 0:
            return rotation_matrix(sour_vertical_vec, np.pi)
        return np.identity(3)
    alpha = np.arccos(np.dot(sour_vec, dest_vec))
    a = np.cross(sour_vec, dest_vec) / np.linalg.norm(np.cross(sour_vec, dest_vec), 2)
    c = np.cos(alpha)
    s = np.sin(alpha)
    R1 = [
        a[0] * a[0] * (1.0 - c) + c,
        a[0] * a[1] * (1.0 - c) - s * a[2],
        a[0] * a[2] * (1.0 - c) + s * a[1],
    ]

    R2 = [
        a[0] * a[1] * (1.0 - c) + s * a[2],
        a[1] * a[1] * (1.0 - c) + c,
        a[1] * a[2] * (1.0 - c) - s * a[0],
    ]

    R3 = [
        a[0] * a[2] * (1.0 - c) - s * a[1],
        a[1] * a[2] * (1.0 - c) + s * a[0],
        a[2] * a[2] * (1.0 - c) + c,
    ]

    R = np.matrix([R1, R2, R3])

    return R


def rotation_matrix(axis, theta):

    # Return the rotation matrix associated with counterclockwise rotation about the given axis by theta radians.

    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.matrix(
        np.array(
            [
                [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
            ]
        )
    )
