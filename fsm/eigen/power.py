from numba import jit
import numpy as np

@jit(nopython=True)
def eig(A, k, method, tol, tol_frac, max_iter, norm_set, norm_iter):
    """Preforms an Eigen Decomposition of Matrix A

    :param A: Binary or Finite CSR Matrix
        if A is not symmetric peig creates A.T and finds
            u = A.T * A * u
    :param k: Integer
        number of eigen-pairs to find
    :param method: String
        {"auto", "single", "block"}
        single -> Power Iteration
        block -> Block Power Iteration
        If method == "auto":
            k <= 2 -> "single"
    :param tol: Float >= 0
        Convergence criteria for when to stop iterating
        if |u_t - u_{t-1}|_{INF} < tol:
            return u_t
    :param tol_frac: Float in (0, 1]
        Switching criteria for when to normalize every iteration
        if tol_frac*|u_t - u_{t-1}|_{INF} < tol:
            norm_iter = 1
            norm_frac = 1
    :param max_iter: Integer > 0
        Maximum number of iterations before return eigen-pairs even
        before tolerance is below designated tolerance

        *Note* will raise an warning

        if iter_num > max_iter:
            return u_t

    :param norm_set: Integer > 1
        Number of contigous sets u is broken into
        ...

        Percentage of u_t used to determine normalization factor.
        Each normalization of u_t will take a contiguous set of
        ceiling of |u|*norm_frac elements of u_t. And will advance
        the set used by the next normalization by |u|*norm_frac.

        Example: norm_frac = k/n
        normalization iter_1:

            [x1, x2, x3, ..., xk, xk+1, xk+2, ..., xn-2, xn-1, xn]
            \__________________/
            Used to estimate normalization

        normalization iter_2:

            [x1, ..., xk, xk+1, xk+2, ..., x2k, x2k+1, ..., xn]
                          \______________/
            Used to estimate normalization

    :param norm_iter: Integer > 0
        Number of iterations between each normalization.

        *Note* is changed by tol_frac when near convergence.

    :return:
    """
    return None


@jit(nopython=True)
def peig(array, k, tol, tol_frac, max_iter=10000, norm_iter=1):
    m, n = array.shape
    if m != n:
        raise Exception('Only Symmetrical Matrices')

    eig_vec = np.zeros((k, m))  # C_CONTIGUOUS arrays therefor a row is an eigenvector
    eig_val = np.zeros(k)

    for i in range(k):
        iter_num = 0
        i_norm_iter = norm_iter
        u_p1 = np.random.randn(n)
        res = np.inf

        while res >= tol:
            u = u_p1
            u_p1 = array.dot1d(u_p1)

            for j in range(0, i):
                # Bx = Ax - lambda_1*<u_1',u_1>*x - lambda_2*<u_2',u_2>*x - ... - lambda_i*<u_i',u_i>*x
                u_p1 -= eig_val[j] * (eig_vec[j, :].dot(u)) * eig_vec[j, :]

            iter_num += 1

            if iter_num > max_iter:
                u_p1 /= np.linalg.norm(u_p1, 2)
                break

            if iter_num % i_norm_iter <= 1:
                # Only Normalize every:
                # norm_iter and norm_iter + 1 iterations to allow for an accurate residual calculation

                u_p1 /= np.linalg.norm(u_p1, 2)
                res = np.linalg.norm(u - u_p1, np.inf)

                if tol_frac*res <= tol:
                    # When within tol_frac of specified tolerance
                    # Normalize every iteration.
                    i_norm_iter = 1

        eig_vec[i, :] = u_p1
        eig_val[i] = array.dot1d(u_p1).dot(u_p1)

    return eig_val, eig_vec.T

@jit(nopython=True)
def beig(array, k, tol, tol_frac, max_iter=10000, norm_iter=3, buffer=10):
    m, n = array.shape
    if m != n:
        raise Exception('Only Symmetrical Matrices')

    vp1 = np.random.randn(m, k+buffer)
    vp1 = np.asfortranarray(vp1)
    res = np.inf
    iter_num = 0

    while res >= tol:
        v = vp1
        vp1 = array.dot2d(v)

        iter_num += 1

        if iter_num > max_iter:
            vp1, _ = np.linalg.qr(vp1)
            break

        if iter_num % norm_iter <= 1:
            # Only Normalize every:
            # norm_iter and norm_iter + 1 iterations to allow for an accurate residual calculation
            vp1, _ = np.linalg.qr(vp1)
            # Over just k not (k + buffer)
            # or Use SVD on (n by k)
            res = np.linalg.norm(v - vp1, np.inf)

            if tol_frac * res <= tol:
                # When within tol_frac of specified tolerance
                # Normalize every iteration.
                norm_iter = 1

    return np.diag(array.dot2d(vp1).T.dot(vp1))[:k], vp1[:, :k]