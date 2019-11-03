import numpy as np
from numba import jit
from .spartype import *


@jit(nopython=True)
def SNP(m, n, max_density, min_density=0, data_n=1, sym=False):
    """
    Creates a sparse M by N matrix following the procedure:

    A = [A1, A2, ..., AN] st.

    prob(Aj) ~ U[min_density, max_density]
    A[i, j] ~ Binomial(Data, prob(Aj))

    Where data is n and prob(Aj) is p in
        X ~ B(n, p)


    However, this is not symmetrical so we adjust to:

    A = [0, A2, A3, ..., AN;
         0,  0, A3, ..., AN;
         0,  0, 0,  ..., AN;
         .
         0,  0, 0,  ..., AN;
         0,  0, 0,  ..., 0]

    A[i, j] ~ Binomial(Data_Range, prob(Aj)) if i < j, else 0
    A <- A + A'

    A[i,i] ~ Binomial(Data_Range, prob(Ai))

    :param m: Number of Rows
    :param n: Number of Columns
    :param max_density: Maximum density of a column
    :param min_density: Minimum density of a column
    :param data_n: Generation of
    """
    m = np.int64(m)
    n = np.int64(n)

    binomials = np.random.uniform(min_density, max_density, size=n)
    rows = []
    cols = []
    data = []

    for j in range(n):
        b_temp = binomials[j]
        if sym:
            rng_max = min(j, m) # Only fill triangle without diagonal
        else:
            rng_max = m
        for i in range(0, rng_max):
            data_temp = np.random.binomial(data_n, b_temp)
            if data_temp > 0:
                # A[i,j] -> A
                rows.append(i)
                cols.append(j)
                data.append(data_temp)
                if sym:
                    # A[j, i] -> A transpose
                    rows.append(j)
                    cols.append(i)
                    data.append(data_temp)
        # A[j, j] -> Diag(A)
        if sym:
            data_temp = np.random.binomial(data_n, b_temp)
            if data_temp > 0:
                rows.append(j)
                cols.append(j)
                data.append(data_temp)

    shape = (m, n)
    data_array = np.array(data).astype(FLOAT_STORAGE_np)
    rows_array = np.array(rows).astype(INDEX_STORAGE_np)
    cols_array = np.array(cols).astype(INDEX_STORAGE_np)

    return data_array, rows_array, cols_array, shape
