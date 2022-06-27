import unittest
import numpy as np

from sor_iteration import SparseMatrix, sor


class SORIterationTest(unittest.TestCase):

    def test_sparse_matrix_manipulation(self):
        n = 5
        np_arr = np.zeros((n, n))
        sparse_arr = SparseMatrix(n)

        indices = [(0, 2), (3, 4), (1, 1), (3, 2), (0, 4), (2, 2), (4, 0), (3, 1), (0, 0), (4, 4), (0, 2), (1, 1), (2, 2)]

        for idx, (i, j) in enumerate(indices, 1):
            np_arr[i, j] = idx
            sparse_arr[i, j] = idx

        sparse_arr_np = sparse_arr.to_np()
        assert np.array_equal(np_arr, sparse_arr_np)
        assert np_arr[1, 1] == sparse_arr[1, 1]
        assert np_arr[0, 3] == sparse_arr[0, 3]
        assert np_arr[4, 1] == sparse_arr[4, 1]

    def test_sparse_matrix_min_space_consumption(self):
        n = 5
        np_arr = np.zeros((n, n))
        sparse_arr = SparseMatrix(n)
        indices = [(i, i) for i in range(n)]

        for idx, (i, j) in enumerate(indices, 1):
            np_arr[i, j] = idx
            sparse_arr[i, j] = idx

        assert sparse_arr.I.shape[1] == 1
        assert sparse_arr.V.shape[1] == 1

    def test_sparse_matrix_numpy_conversion(self):
        np_arr = np.zeros((5, 5))

        indices = [(0, 2), (3, 4), (1, 1), (3, 2), (0, 4), (2, 2), (4, 0), (3, 1), (0, 0), (4, 4), (0, 2), (1, 1), (2, 2)]
        for idx, (i, j) in enumerate(indices, 1):
            np_arr[i, j] = idx

        assert np.array_equal(np_arr, SparseMatrix.from_np(np_arr).to_np())

    def test_sparse_matrix_matmul(self):

        np_arr = np.zeros((5, 5))

        indices = [(0, 2), (3, 4), (1, 1), (3, 2), (0, 4), (2, 2), (4, 0), (3, 1), (0, 0), (4, 4), (0, 2), (1, 1), (2, 2)]
        for idx, (i, j) in enumerate(indices, 1):
            np_arr[i, j] = idx

        sparse_mat = SparseMatrix.from_np(np_arr)

        vec = np.arange(5)

        res_np = np_arr @ vec
        res_sparse = sparse_mat @ vec

        assert np.array_equal(res_np, res_sparse)

    def test_sparse_matrix_sor(self):
        A = np.array([[10., 4., 0, 0.],
                      [2., 11., -1., 0],
                      [0, -2., 10., -3.],
                      [0., 0, 6, 8.]])

        x = np.arange(4) * 2 + 1
        b = A @ x

        x_sol, _ = sor(SparseMatrix.from_np(A), b, np.random.random(size=4), 0.8)

        assert np.allclose(x, x_sol)


