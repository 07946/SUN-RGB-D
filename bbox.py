import numpy as np


class Bbox3d:
    def __init__(self, basis, coeffs, centroid, **kwargs):
        self.basis = basis
        self.coeffs = coeffs
        self.centroid = centroid
        self.className = kwargs['className'] if 'className' in kwargs.keys() else None
        self.sequenceName = kwargs['sequenceName'] if 'sequenceName' in kwargs.keys() else None
        self.orientation = kwargs['orientation'] if 'orientation' in kwargs.keys() else None
        self.label = kwargs['label'] if 'label' in kwargs.keys() else None
        self.corner = np.zeros((8, 3), dtype=np.float32)

    def getCorner(self):
        if np.any(self.corner != 0):
            return self.corner
        corner = np.zeros_like(self.corner)
        coeffs = self.coeffs.ravel()
        indices = np.argsort(- np.abs(self.basis[:, 0]))
        basis = self.basis[indices, :]
        coeffs = coeffs[indices]
        indices = np.argsort(- np.abs(basis[1:3, 1]))
        if indices[0] == 1:
            basis[[1, 2], :] = basis[[2, 1], :]
            coeffs[[1, 2]] = coeffs[[2, 1]]

        basis = flip_toward_viewer(basis, np.repeat(self.centroid, 3, axis=0))
        coeffs = abs(coeffs)
        corner[0] = -basis[0] * coeffs[0] + basis[1] * coeffs[1] + basis[2] * coeffs[2]
        corner[1] = basis[0] * coeffs[0] + basis[1] * coeffs[1] + basis[2] * coeffs[2]
        corner[2] = basis[0] * coeffs[0] + -basis[1] * coeffs[1] + basis[2] * coeffs[2]
        corner[3] = -basis[0] * coeffs[0] + -basis[1] * coeffs[1] + basis[2] * coeffs[2]

        corner[4] = -basis[0] * coeffs[0] + basis[1] * coeffs[1] + -basis[2] * coeffs[2]
        corner[5] = basis[0] * coeffs[0] + basis[1] * coeffs[1] + -basis[2] * coeffs[2]
        corner[6] = basis[0] * coeffs[0] + -basis[1] * coeffs[1] + -basis[2] * coeffs[2]
        corner[7] = -basis[0] * coeffs[0] + -basis[1] * coeffs[1] + -basis[2] * coeffs[2]
        corner += np.repeat(self.centroid, 8, axis=0)
        self.corner = corner
        return corner


def flip_toward_viewer(normals, points):
    points /= np.linalg.norm(points, axis=1)
    projection = np.sum(points * normals, axis=1)
    flip = projection > 0
    normals[flip] = - normals[flip]
    return normals
