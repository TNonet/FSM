
def peig(A, k, method, tol, tol_frac, max_iter, norm_frac, norm_iter):
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

    :param norm_frac: Float in (0, 1]
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

