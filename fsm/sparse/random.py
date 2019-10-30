import numpy as np
import numba as nb
from .spartype import *

@jit(nopython=True)
def SNP(m, n, max_density, min_density=0, data_n=1):
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
    """
    m = np.int64(m)
    n = np.int64(n)

    binomials = np.random.uniform(min_density, max_density, size=n)
    rows = []
    cols = []
    data = []

    for j in range(n):
        b_temp = binomials[j]
        for i in range(0, min(j, m)):
            data_temp = np.random.binomial(data_n, b_temp)
            if data_temp > 0:
                # A[i,j] -> A
                rows.append(i)
                cols.append(j)
                data.append(data_temp)
                # A[j, i] -> A transpose
                rows.append(j)
                cols.append(i)
                data.append(data_temp)
        # A[j, j] -> Diag(A)
        data_temp = np.random.binomial(data_n, b_temp)
        if data_temp > 0:
            rows.append(j)
            cols.append(j)
            data.append(data_temp)

    shape = (m, n)
    data_array = np.array(data)
    rows_array = np.array(rows)
    cols_array = np.array(cols)

    return data_array, rows_array, cols_array, shape