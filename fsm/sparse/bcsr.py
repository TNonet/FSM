"""Compressed Binary Sparse Row Matrix"""

import numba as nb
import numpy as np

from .spartype import *
from .matmul import binary_matmul_1d, binary_matmul_2d
from .transform import coo_to_csr

def bcsr_constructor(shape=None, data=None):
    """
    Notes:
        numba requires all functions to return consistent types.
    :return: 
    """

    if shape is None and data is None:
        shape = (4, 4)
        row_p = np.array([0, 1, 2, 3, 4]).astype(POINT_STORAGE_np)
        col_i = np.array([0, 1, 2, 3]).astype(INDEX_STORAGE_np)
        return bcsr_matrix(row_p, col_i, shape)

    #Logic for Determining;
    #   1. Feasibility of Data
    #   2. Converting Data to Proper Format
    #   3. Changing Types of Data

    m, n = shape
    rows, cols = data
    nnz = len(rows)

    row_p, col_i = coo_to_csr(m, nnz, rows, cols, data=None)
    return bcsr_matrix(row_p, col_i, shape)


bcsr_spec = [
    ('shape', nb.types.UniTuple(nb.int64, 2)),
    ('row_p', POINT_STORAGE_nb[:]),
    ('col_i', INDEX_STORAGE_nb[:]),
]

@nb.jitclass(bcsr_spec)
class bcsr_matrix:
    def __init__(self, _row_p, _col_i, _shape):
        """
        :param _row_p: np.array of integers
        :param _col_i: np.array of integers
        :param _shape: 2-tuple of integers (m, n) representing shape of Matrix

        See Documentation for Format
        """

        self.shape = _shape
        self.row_p = _row_p
        self.col_i = _col_i

    def dot1d(self, other):
        """Matrix-Vector Dot Product

        :param other:
        :return:
        """
        m, n = self.shape

        if len(other.shape) == 2:
            raise NameError("Use dot2d")
        elif len(other.shape) > 2:
            raise NotImplementedError

        d1 = other.shape[0]
        m, n = self.shape

        if n != d1:
            raise Exception('Dimension MisMatch')

        out = np.zeros(n, dtype=FLOAT_STORAGE_np)
        binary_matmul_1d(out, self.row_p, self.col_i, other, m)
        return out

    def dot2d(self, other):
        """Matrix-Matrix Dot Product
        :param other: Fortran Stored Array
        :return:
        """
        m, n = self.shape

        if len(other.shape) == 1:
            raise NameError("Use dot1d")
        elif len(other.shape) > 2:
            raise NotImplementedError

        if other.flags.c_contiguous:
            raise ValueError("Use Fortran Array")

        d1, k = other.shape

        if n != d1:
            raise Exception('Dimension MisMatch')

        out = np.zeros((n, k), dtype=FLOAT_STORAGE_np)
        out = np.asfortranarray(out)
        binary_matmul_2d(out, self.row_p, self.col_i, other, m, k)
        return out

    # def __str__(self):
    #     return self.to_array()
    #
    # def to_array(self):
    #     _rows, _cols = csr_to_coo(self.row_p, self.col_i, self.shape)
    #     return coo_to_array(_rows, _cols, shape=self.shape)
    #
    # @property
    # def size(self):
    #     return len(self.col_i)
    #
    # @property
    # def sparsity(self):
    #     return self.size / (self.shape[0] * self.shape[1])
    #
    # def __sizeof__(self):
    #     """
    #     returns roughly the memory storage of instance in Bytes
    #
    #     Storage Includes:
    #         self.row_p
    #         self.col_i
    #
    #     :return:
    #     """
    #
    #     n_bytes = 4 * (len(self.col_i) + len(self.row_p))
    #
    #     mem_dict = {'bytes': np.float64(n_bytes)}
    #
    #     if n_bytes < 2 ** 10:
    #         mem_dict['bytes'] = np.float64(n_bytes)
    #     elif n_bytes < 2 ** 20:
    #         mem_dict['"kilobytes"'] = np.round(np.float64(n_bytes) / 2 ** 10, 2)
    #     elif n_bytes < 2 ** 30:
    #         mem_dict["megabytes"] = np.round(np.float64(n_bytes) / 2 ** 20, 2)
    #     elif n_bytes < 2 ** 40:
    #         mem_dict['gigabytes'] = np.round(np.float64(n_bytes) / 2 ** 30, 2)
    #
    #     return mem_dict
    #
    # @property
    # def T(self):
    #     m, n = self.shape
    #     rows, cols = csr_to_coo(self.row_p, self.col_i, self.shape)
    #     return BinarySparse(cols, rows, np.array([n, m])

