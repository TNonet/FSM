from fsm.sparse.bcsr import bcsr_matrix
from fsm.sparse.fcsr import fcsr_constructor
from fsm.sparse.random import SNP
from fsm.sparse.transform import coo_to_csr, coo_to_dense

import numpy as np


def test_b_matrix(m=None, n=None, data_n=1):
    while True:
        _m = m or np.random.randint(1, 100)
        _n = n or np.random.randint(1, 100)
        data_array, rows_array, cols_array, shape = SNP(_m, _n, .1, data_n=data_n)

        row_p, col_i = coo_to_csr(_m, len(rows_array), rows_array, cols_array)
        array_np = coo_to_dense(rows_array, cols_array, (_m, _n))
        array_sparse = bcsr_matrix(row_p, col_i, (_m, _n))

        yield array_np, array_sparse


def test_f_matrix(m=None, n=None, k=None, data_n=1, density=.1):
    while True:
        _k = k or np.random.randint(1, 10)
        b_decomp = []
        _m = m or np.random.randint(1, 100)
        _n = n or np.random.randint(1, 100)
        for i in range(_k):
            value = np.random.randint(1, 10)
            data_array, rows_array, cols_array, shape = SNP(_m, _n, density, data_n=data_n)
            row_p, col_i = coo_to_csr(_m, len(rows_array), rows_array, cols_array)
            array_sparse = bcsr_matrix(row_p, col_i, (_m, _n))
            array_sparse.alpha = value
            b_decomp.append(array_sparse)

        b_decomp = tuple(b_decomp)

        array_sparse = fcsr_constructor(b_decomp)

        array_np = array_sparse.to_array()

        yield array_np, array_sparse

