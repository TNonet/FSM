"""Compressed Finite Sparse Row Matrix"""

import numba as nb
import numpy as np

from .spartype import *
from .bcsr import bcsr_constructor, bcsr_matrix
from .matmul import finite_matmul_1d, finite_matmul_2d



def fcsr_constructor(num):
    """
    Notes:
        numba requires all functions to return consistent types.


    :return:
    """

    # Logic for Determining;
    #   1. Feasibility of Data
    #   2. Converting Data to Proper Format
    #   3. Changing Types of Data
    binary_decomp = []
    for i in range(1,num):
        binary_decomp.append(bcsr_constructor(i, i))

    binary_decomp = tuple(binary_decomp)

    return fcsr_matrix(binary_decomp, len(binary_decomp), (5,5))

bcsr_type = nb.deferred_type()
bcsr_type.define(bcsr_matrix.class_type.instance_type)

fcsr_spec = [
    ('shape', nb.types.UniTuple(nb.int64, 2)),
    ('depth', nb.int64),
    ('b_decomp', nb.types.UniTuple(bcsr_type, 4)),
]


@nb.jitclass(fcsr_spec)
class fcsr_matrix:
    def __init__(self, _b_decomp, _depth, _shape):
        """

        :param _row_p:
        :param _col_i:
        :param _shape:
        """

        self.shape = _shape
        self.depth = _depth
        self.b_decomp = _b_decomp

    def dot1d(self, other):
        """

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
        finite_matmul_1d(out, self.b_decomp, other)
        return out

    def dot2d(self, other):
        """

        :param other:
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
        finite_matmul_2d(out, self.b_decomp, other)
        return out
