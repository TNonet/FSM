import numpy as np
import numba as nb

from .spartype import *

@nb.jit(nopython=True)
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
def coo_to_array(rows_indices, col_indices, shape):
    array = np.zeros((np.int64(shape[0]), np.int64(shape[1])))
    for i, j in zip(rows_indices, col_indices):
        array[i, j] = 1
    return array.astype(d_type)