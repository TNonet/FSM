"""Compressed Finite Sparse Row Matrix"""

from numba import jitclass
import numpy as np

from .spartype import *
from .bcsr import bcsr_matrix
from .matmul import finite_matmul_2d, finite_matmul_1d


def fcsr_constructor(b_matrix_iterable):
    """
    Notes:
        numba requires all functions to return consistent types.


    :return:
    """

    # Logic for Determining;
    #   1. Feasibility of Data
    #   2. Converting Data to Proper Format
    #   3. Changing Types of Data
    shape = None
    for b_matrix in b_matrix_iterable:
        if shape is None:
            shape = b_matrix.shape
        elif shape == b_matrix.shape:
            pass
        else:
            raise ValueError("Shapes don't Match")

    depth = len(b_matrix_iterable)
    b_matrix_decomp = tuple(b_matrix_iterable)

    bcsr_type = nb.deferred_type()
    bcsr_type.define(bcsr_matrix.class_type.instance_type)

    fcsr_spec = [
        ('shape', nb.types.UniTuple(nb.int64, 2)),
        ('b_decomp', nb.types.UniTuple(bcsr_type, depth)),
    ]

    @jitclass(fcsr_spec)
    class fcsr_matrix:
        def __init__(self, _b_decomp, _shape):
            """

            :param _row_p:
            :param _col_i:
            :param _shape:
            """

            self.shape = _shape
            self.b_decomp = _b_decomp

        @property
        def depth(self):
            return len(self.b_decomp)

        def dot1d(self, other):
            """

            :param other:
            :return:
            """
            if len(other.shape) != 1:
                raise Exception('Must be a 1d Array')

            d1 = other.shape[0]
            m, n = self.shape

            if n != d1:
                raise Exception('Dimension MisMatch')

            return finite_matmul_1d(self.b_decomp, other, m)

        def dot2d(self, other):
            """

            :param other:
            :return:
            """
            if len(other.shape) != 2:
                raise Exception('Must be a 2d Array')

            m, n = self.shape
            d1, k = other.shape

            # if k > 1 and other.flags.c_contiguous:
            #     raise Exception("Use Fortran Array")

            if n != d1:
                raise Exception('Dimension MisMatch')

            return finite_matmul_2d(self.b_decomp, other, m, k)

        def to_array(self):
            array = np.zeros(self.shape)
            for sub_b_matrix in self.b_decomp:
                array += sub_b_matrix.to_array()
            return array

        @property
        def size(self):
            nnz = 0
            for sub_b_matrix in self.b_decomp:
                nnz += sub_b_matrix.size
            return nnz

        @property
        def sparsity(self):
            return self.size / (self.shape[0] * self.shape[1])

        @property
        def mem(self):
            return self.__sizeof__()

        def __sizeof__(self):
            """
            returns roughly the memory storage of instance in Bytes

            Storage Includes:
                self.row_p
                self.col_i

            :return:
            """
            mem = 0
            for b_matrix in self.b_decomp:
                mem += b_matrix.mem
            return mem

    return fcsr_matrix(b_matrix_decomp, shape)
