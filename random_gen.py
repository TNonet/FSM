from fsm.sparse.bcsr import bcsr_matrix
from fsm.sparse.fcsr import fcsr_constructor
from fsm.sparse.random import SNP
from fsm.sparse.transform import coo_to_csr, coo_to_dense
from fsm.sparse.spartype import FLOAT_STORAGE_np

import numpy as np
from scipy.sparse.csr import csr_matrix

def test_b_matrix(m=None, n=None, density=.1, data_n=1, sym=False, return_np=True):
    while True:
        if sym:
            _m = _n = m or np.random.randint(1, 100)
        else:
            _m = m or np.random.randint(1, 100)
            _n = n or np.random.randint(1, 100)
        rows_array, cols_array, data_array, shape = SNP(_m, _n, density, data_n=data_n, sym=sym)

        row_p, col_i = coo_to_csr(_m, len(rows_array), rows_array, cols_array)

        if return_np:
            array_np = coo_to_dense(rows_array, cols_array, (_m, _n))
            array_sparse = bcsr_matrix(row_p, col_i, (_m, _n))
            array_scipy = csr_matrix((data_array, (rows_array, cols_array)), shape)
            yield array_np, array_sparse, array_scipy
        else:
            array_sparse = bcsr_matrix(row_p, col_i, (_m, _n))
            array_scipy = csr_matrix((data_array, (rows_array, cols_array)), shape)
            yield array_sparse, array_scipy



def test_f_matrix(m=None, n=None, k=None, data_n=1, density=.1, sym=False, return_np=True):
    while True:
        _k = k or np.random.randint(1, 10)
        b_decomp = []
        if sym:
            _m = _n = m or np.random.randint(1, 100)
        else:
            _m = m or np.random.randint(1, 100)
            _n = n or np.random.randint(1, 100)

        shape = (_m, _n)
        array_scipy = csr_matrix(shape, dtype=FLOAT_STORAGE_np)
        for i in range(_k):
            value = i+1
            rows_array, cols_array, data_array, shape = SNP(_m, _n, density, data_n=data_n, sym=sym)
            array_scipy += csr_matrix((value*data_array, (rows_array, cols_array)), shape)
            row_p, col_i = coo_to_csr(_m, len(rows_array), rows_array, cols_array)
            array_sparse = bcsr_matrix(row_p, col_i, shape)
            array_sparse.alpha = value
            b_decomp.append(array_sparse)

        b_decomp = tuple(b_decomp)

        array_sparse = fcsr_constructor(b_decomp)

        if return_np:
            array_np = array_sparse.to_array()
            yield array_np, array_sparse, array_scipy
        else:
            yield array_sparse, array_scipy

