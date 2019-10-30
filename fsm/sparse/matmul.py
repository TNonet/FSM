import numba as nb

@nb.jit(nopython=True)
def binary_matmul_1d(out, row_p, col_i, other, m):
    """Matrix-Vector Dot Product
    A*v
    A -> (row_p, col_i)
        CSR format (documentation in folder)
        shape: (m, n)
    v -> other
        np.array(ndims=1)
        shape: (n,)

    :param out: [ADD HERE]
    :param row_p: np.array(integer type)
        shape: (m+1,)
        note: map between row to col_i
    :param col_i: np.array(integer type)
        shape: (nnz,), nnz -> number of non-zeros in A
    :param other: np.array(numeric type)
        shape: (n,)
    :param m: integer type
        note: number of columns of A == number of rows of v
    :param k: integer type
        note: number of columns of v

    :return: out: np.array(float64)
        shape: (m, 1)
        format: C
    """
    for i in range(m):
        vector_sum = 0
        for r in col_i[row_p[i]:row_p[i + 1]]:
            vector_sum += other[r]
        out[i] = vector_sum


@nb.jit(nopython=True, parallel=True, nogil=True)
def binary_matmul_2d(out, row_p, col_i, other, m, k):
    for j in nb.prange(k):
        binary_matmul_1d(out[:, j], row_p, col_i, other[:, j], m)


@nb.jit(nopython=True, parallel=True, nogil=True)
def finite_matmul_1d(out, b_decomp, other):
    for b_matrix in b_decomp:
        out += b_matrix.dot1d(other)

@nb.jit(nopython=True)
def finite_matmul_2d(out, b_decomp, other):
    for b_matrix in b_decomp:
        out += b_matrix.dot2d(other)


