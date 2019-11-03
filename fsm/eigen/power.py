from numba import jit
import numpy as np

@jit(nopython=True)
def peig(A, k, method, tol, tol_frac, max_iter, norm_set, norm_iter):
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
def peig(array, k, tol, tol_frac, max_iter, norm_set, norm_iter):
    m, n = array.shape
    if m != n:
        raise Exception('Only Symmetrical Matrices')

    eig_vec = np.zeros((m, k))
    eig_vec = np.asfortranarray(eig_vec)
    eig_val = np.zeros(k)

    norm_setsize = np.int64(np.round(m/norm_set))

    for i in range(k):
        iter_num = 0
        iter_norm_num = 0
        i_norm_iter = norm_iter
        i_norm_set = norm_set
        i_norm_setsize = norm_setsize
        u_p1 = np.random.randn(n)
        res = np.inf

        while res >= tol:
            u = u_p1
            u_p1 = array.dot1d(u_p1)

            for j in range(0, i):
                # Bx = Ax - lambda_1*<u_1',u_1>*x - lambda_2*<u_2',u_2>*x - ... - lambda_i*<u_i',u_i>*x
                u_p1 -= eig_val[j] * (eig_vec[:, j].dot(u)) * eig_vec[:, j]

            iter_num += 1

            if iter_num > max_iter:
                u_p1 /= np.linalg.norm(u_p1, 2)
                break

            # Only Normalize every norm_iter iterations
            if iter_num % i_norm_iter == 0:
                iter_norm_num += 1
                if iter_norm_num % i_norm_set == 0:
                    # If at the end norm_setsize may not divide m with no remainder
                    # Thus we grab the end of u_p1 to make sure we grab all elements
                    norm_factor = i_norm_set*np.linalg.norm(u_p1[m-i_norm_setsize:], 2)
                    u_p1 /= norm_factor
                    iter_norm_num = 0
                else:
                    norm_factor = i_norm_set*np.linalg.norm(
                        u_p1[(iter_norm_num-1)*i_norm_setsize:iter_norm_num*i_norm_setsize],
                        2)
                    u_p1 /= norm_factor

                u /= norm_factor
                print(iter_num)
                print(u)
                print(u_p1)
                res = np.linalg.norm(u - u_p1, np.inf)

                # When within tol_frac of specified tolerance
                # Normalize every iteration with full data.
                if tol_frac*res <= tol:
                    i_norm_iter = 1
                    i_norm_set = 1
                    i_norm_setsize = m

        eig_vec[:, i] = u_p1
        eig_val[i] = array.dot1d(u_p1).dot(u_p1)

    return eig_val, eig_vec

@jit(nopython=True)
def beig(A, k, tol, tol_frac, max_iter, norm_frac, norm_iter):
    return None


@jit(nopython=True)
def get_item_wrap_index_1d(array, rng):
    """
    rng = 2-Tuple
        (Start, Stop) with wrap around

    Cases:
        1) 0 < start < stop < m
            return array[start:stop, :]
        2) 0 < stop < start < m
            return array[start:end] & array[0, stop]
        3) Anything else:
            return np.array([])
    """
    m = array.shape[0]
    if 0 <= rng[0] < rng[1] <= m:
        return array[rng[0]:rng[1]]
    elif 0 <= rng[1] < rng[0] <= m:
        return np.hstack((array[rng[0]:], array[:rng[1]]))
    else:
        return array[0:0]


@jit(nopython=True)
def get_item_wrap_index_2d(array, rng):
    m, k = array.shape

    if 0 <= rng[0] < rng[1] <= m:
        n = rng[1] - rng[0]
    elif 0 <= rng[1] < rng[0] <= m:
        n = m - rng[0] + rng[1]

    out = np.zeros((m, n))
    out = np.asfortranarray(out)
    for i in range(0, k):
        out[i, :] = get_item_wrap_index_1d(array[:, i], rng)
    return out.T