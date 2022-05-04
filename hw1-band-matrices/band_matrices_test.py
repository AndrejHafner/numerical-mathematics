import unittest
import numpy as np
from unittest.mock import MagicMock
from band_matrices import BandMatrix, gen_rand_band_matrix, LowerBandMatrix, UpperBandMatrix


def get_diag_matrices():
    band_width = 5
    mat = np.diag(np.ones(7))
    mat[3, 2] = 1
    mat[4, 3] = 5
    mat[5, 6] = 10
    mat[4, 6] = 8
    mat[5, 3] = -4
    band_mat_1 = mat.copy()
    mat[5, 5] = 22
    mat[1, 0] = 4
    mat[2, 4] = -3
    mat[3, 5] = -8
    mat[5, 6] = -1
    band_mat_2 = mat.copy()
    return band_mat_1, band_mat_2, band_width


class BandMatricesTest(unittest.TestCase):

    def test_numpy_conversion(self):
        mat, _, band_width = get_diag_matrices()
        band_mat = BandMatrix.from_np_matrix(mat, band_width)
        assert np.array_equal(mat, band_mat.to_np_matrix())

    def test_basic_operations(self):
        mat1, mat2, band_width = get_diag_matrices()
        band1, band2 = BandMatrix.from_np_matrix(mat1, band_width),  BandMatrix.from_np_matrix(mat2, band_width)

        band_sum = band1 + band2
        assert np.array_equal(band_sum.to_np_matrix(), mat1 + mat2)

        band_sub = band1 - band2
        assert np.array_equal(band_sub.to_np_matrix(), mat1 - mat2)

        band_mult = band1 * band2
        assert np.array_equal(band_mult.to_np_matrix(), mat1 * mat2)

        band_mult_scalar = band1 * 5
        assert np.array_equal(band_mult_scalar.to_np_matrix(), mat1 * 5)

        band_mult_scalar_r = 2 * band1
        assert np.array_equal(band_mult_scalar_r.to_np_matrix(), mat1 * 2)

        band_div_scalar = band1 / 5
        assert np.array_equal(band_div_scalar.to_np_matrix(), mat1 / 5)

        band_div_scalar_r = 5 / band1
        assert np.array_equal(band_div_scalar_r.to_np_matrix(), mat1 / 5)

    def test_matmul(self):
        mat1, mat2, band_width = get_diag_matrices()
        band1, band2 = BandMatrix.from_np_matrix(mat1, band_width), BandMatrix.from_np_matrix(mat2, band_width)
        vec = np.arange(7)
        res1 = band1 @ vec
        res_true = mat1 @ vec
        assert np.array_equal(res1, res_true)

    def test_matmul_lower(self):
        n = 9
        band_width = 5
        matrix = gen_rand_band_matrix(n, band_width)
        band_matrix = BandMatrix.from_np_matrix(matrix, band_width)
        lower_mat = LowerBandMatrix.from_diags(band_matrix.main_diag, band_matrix.lower_diags)

        vec = np.arange(9)
        res = lower_mat.to_np_matrix() @ vec
        res_band = lower_mat @ vec
        assert np.array_equal(res, res_band)

    def test_matmul_upper(self):
        n = 9
        band_width = 5
        matrix = gen_rand_band_matrix(n, band_width)
        band_matrix = BandMatrix.from_np_matrix(matrix, band_width)
        lower_mat = UpperBandMatrix.from_diags(band_matrix.main_diag, band_matrix.upper_diags)

        vec = np.arange(9)
        res = lower_mat.to_np_matrix() @ vec
        res_band = lower_mat @ vec
        assert np.array_equal(res, res_band)

    def test_lu(self):
        np.random.seed(4)
        n = 20
        for band_width in range(1, 18, 2):
            matrix = gen_rand_band_matrix(n, band_width)
            band_matrix = BandMatrix.from_np_matrix(matrix, band_width)
            band_matrix.is_diagonally_dominant = MagicMock(return_value=True)
            L, U = band_matrix.lu()
            assert np.allclose(matrix, L.to_np_matrix() @ U.to_np_matrix())

    def test_left_divide(self):
        mat = gen_rand_band_matrix(20, 13)
        band_mat = BandMatrix.from_np_matrix(mat, 13)
        band_mat.is_diagonally_dominant = MagicMock(return_value=True)
        x = np.random.randint(1, 20, 20)
        b = mat @ x
        x_sol = band_mat.left_divide(b)
        assert np.allclose(x, x_sol)

    def test_left_divide_lower(self):
        np.random.seed(42)
        mat = gen_rand_band_matrix(20, 13)
        band_mat = BandMatrix.from_np_matrix(mat, 13)
        lower_mat = LowerBandMatrix.from_diags(band_mat.main_diag, band_mat.lower_diags)
        lower_mat.is_diagonally_dominant = MagicMock(return_value=True)
        x = np.random.randint(1, 20, 20)
        b = lower_mat @ x
        x_sol = lower_mat.left_divide(b)
        assert np.allclose(x, x_sol)

    def test_left_divide_upper(self):
        np.random.seed(42)
        mat = gen_rand_band_matrix(20, 13)
        band_mat = BandMatrix.from_np_matrix(mat, 13)
        upper_mat = UpperBandMatrix.from_diags(band_mat.main_diag, band_mat.upper_diags)
        upper_mat.is_diagonally_dominant = MagicMock(return_value=True)
        x = np.random.randint(1, 20, 20)
        b = upper_mat @ x
        x_sol = upper_mat.left_divide(b)
        assert np.allclose(x, x_sol)


if __name__ == '__main__':
    unittest.main()