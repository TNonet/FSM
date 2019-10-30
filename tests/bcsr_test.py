import unittest
from fsm.sparse.bcsr import

class BinaryTestCase(unittest.TestCase):
    """Tests
    1. Identity
        a) Variable Shapes (10, 100)
    2. Random
        100x) Variable Shapes (m, n) from SNP
    3. Vector Dot Product
        a) Square (n, n) * (n,)
        b) Non-Square (m, n) * (n,)
        c) [Failure] (n, k) * (n!=k,)
        d) [Failure] (m, k) * (n!=k,)
        e) [Failure] (m, n) * (n, k>1)
        f) [Failure] (m, n) * (n, k1>1, k2>1)
    4. Matrix Dot Product
        a) Square (n, n) * (n, 1)
        b) Square (n, n) * (n, k)
        c) Non-Square (m, n) * (n, 1)
        d) Non-Square (m, n) * (n, k)
        c) [Failure] (m, n) * (k1, k2)
    5. Transpose
        a) One Transpose Check with Numpy
        b) Double Transpose check with self and numpy
    """
    def test_identity(self):
        for i in range(10):
            N = 2**i
            A_np = np.eye(N)
            rows, cols, _ = dense_to_coo(A_np)
            A_sparse = BinarySparse(rows, cols, np.array([N, N]).astype(np.uint32))

            np.testing.assert_array_equal(A_np, A_sparse.to_array())

    def test_numpy_random(self):
        matrix_generator = generate_random_martix()
        for i in range(10):
            A_np, A_sparse = next(matrix_generator)
            np.testing.assert_array_equal(A_np, A_sparse.to_array())

    def test_vect_dot(self):
        matrix_generator = generate_random_martix()
        for i in range(10):
            A, A_sparse = next(matrix_generator)
            m, n = A.shape
            u = np.random.normal(size=n)
            np.testing.assert_array_almost_equal(A@u, A_sparse.dot(u))

    def test_mat_dot(self):
        matrix_generator = generate_random_martix()
        for i in range(10):
            A_np, A_sparse = next(matrix_generator)
            m, n = A_np.shape
            k = np.random.randint(1, 4)
            u = np.random.randn(n, k)
            np.testing.assert_array_almost_equal(A_np@u, A_sparse.dot(u))

    def test_transpose(self):
        matrix_generator = generate_random_martix()
        for i in range(10):
            A_np, A_sparse = next(matrix_generator)
            np.testing.assert_array_al

if __name__ == '__main__':
    unittest.main()
