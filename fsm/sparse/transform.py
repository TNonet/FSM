import numpy as np
from numba import jit, prange

from .spartype import *

@jit(nopython=True)
def coo_to_csr(n_row, nnz, rows_indices, col_indices, data=None):
    """
    Converst data in COO form:
        rows = [x(1), x(2), ..., x(nnz)]
        cols = [y(1), y(2,, ..., y(nnz)]

        Where A[x(i), y(i)] == data(i)

    To CSR:
    col_i:
        np.array of length nnz

        The non_zero indices of the rows of a matrix put into a line.

        A = 1 0 0 0 --> [0]
            1 1 1 0 --> [0, 1, 2]
            0 0 0 1 --> [3]
            0 0 1 0 --> [2]

        col_i = [0, 0, 1, 2, 3, 2]

    row_p:
        np.array of length m+1

        The location in flat_non_zero_list to find the kth row indicies

        A = 1 0 0 0 --> [0]
            1 1 1 0 --> [0, 1, 2]
            0 0 0 1 --> [3]
            0 0 1 0 --> [2]

        row_p = [0, 1, 4, 5, 6]


    To get first kth row non_zero entries:
    k_non_zero = col_i[row_p[k]:row_p[k+1]]
    """

    row_p = np.zeros(n_row + 1).astype(POINT_STORAGE_np)
    col_i = np.zeros(nnz).astype(INDEX_STORAGE_np)

    for i in range(0, nnz):
        row_p[rows_indices[i] + 1] += 1

    row_p = np.cumsum(row_p).astype(POINT_STORAGE_np)
    row_p[n_row] = nnz

    col_index_counter = row_p.copy()

    for i in range(0, nnz):
        row = rows_indices[i]
        col = col_indices[i]

        ix = col_index_counter[row]

        col_i[ix] = col

        col_index_counter[row] += 1

    return row_p, col_i

@jit(nopython=True)
def csr_to_coo(row_p, col_i, shape):
    m, n = shape
    rows_indices = np.zeros(row_p[m]).astype(INDEX_STORAGE_np)
    col_indices = np.zeros_like(rows_indices)
    counter = 0
    for i in range(m):
        for j in col_i[row_p[i]:row_p[i+1]]:
            rows_indices[counter] = i
            col_indices[counter] = j
            counter += 1

    return rows_indices, col_indices

@jit(nopython=True)
def coo_to_dense(rows_indices, col_indices, shape):
    array = np.zeros((np.int64(shape[0]), np.int64(shape[1])))
    for i, j in zip(rows_indices, col_indices):
        array[i, j] = 1
    return array


@jit(nopython=True)
def dense_to_coo(array):
    """

    :param array:
    :return:
    """
    if len(array.shape) != 2:
        raise NotImplementedError("Must be a two dimensional array")
    m, n = array.shape
    rows = []
    cols = []
    data = []
    for i in range(m):
        for j in range(n):
            if array[i, j] != 0:
                rows.append(i)
                cols.append(j)
                data.append(array[i, j])
    return np.array(rows), np.array(cols), np.array(data), (m, n)


@jit(nopython=True, parallel=True, nogil=True)
def TSQR(array, clusters=8):
    """
    https://web.stanford.edu/group/ctr/Summer/SP14/08_Transition_and_turbulence/08_sayadi.pdf
    :param array:
    :param clusters:
    :return:
    """
    m, n = array.shape
    stack_R = np.zeros((clusters*n, n))
    stack_Q = []

    stride_list = [0]
    stride = m//clusters
    curr_stride = stride
    for i in range(clusters-1):
        stride_list.append(curr_stride)
        curr_stride += stride
    stride_list.append(min(m, curr_stride))

    for i in prange(clusters):
        Q, R = np.linalg.qr(array[stride_list[i]:stride_list[i+1], :])
        stack_R[i*n:(i+1)*n, :] = R
        stack_Q.append(Q)

    Q, _ = np.linalg.qr(stack_R)

    return_Q = np.zeros_like(array)
    for i in range(clusters):
        return_Q[stride_list[i]:stride_list[i+1], :] = stack_Q[i].dot(Q[i*n:(i+1)*n, :])
        # print('Return Q: ', return_Q[stride_list[i]:stride_list[i+1], :].shape)
        # print('Stack Q: ', stack_Q[i].shape)
        # print('Middle Q:', Q[i*n:(i+1)*n, :].shape)
    return return_Q


def coo_to_fcoo(rows_indices, col_indices, data, shape):
    """
    Convert general COO form to Finite COO form
        Finite COO:
            {a_1: [row_indicies(a_1), col_indices(a_1)],
            a_2: [row_indicies(a_2), col_indices(a_2)],
            ...
            a_d: [row_indicies(a_d), col_indices(a_d)]}

        Allows for creating d BCSR matrices

    :param rows_indices: Integer np.array \in {1, 2, ... m}^N
    :param col_indices: Integer np.array \in {1, 2, ..., n}^N
    :param data: Numeric np.array \in {a_1, a_2, ... a_d}^N
    :param shape: 2-Tuple of Intergers (m, n)
    """
    m, n = shape
    if 1 > m or 1 > n:
        raise Exception('Matrix must have positive dimensions')

    Nnz = len(data)

    if Nnz != len(col_indices):
        raise Exception
    if Nnz != len(rows_indices):
        raise Exception

    if m * n < Nnz:
        raise Exception("Too many elements to store")
    if m*n < 10 * Nnz:
        raise Warning("Matrix has sparsity above 10%")

    fcoo_dict = {}
    for i in range(Nnz):
        if data[i] not in fcoo_dict:
            fcoo_dict[data[i]] = [[rows_indices[i]], [col_indices[i]]]
        else:
            fcoo_dict[data[i]][0].append(rows_indices[i])
            fcoo_dict[data[i]][1].append(col_indices[i])

    return fcoo_dict






