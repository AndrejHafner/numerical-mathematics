import unittest
from normal_cdf import normal_cdf


class TestNormalCDF(unittest.TestCase):

    def test_normal_cdf_half(self):
        eps = 1e-10
        assert normal_cdf(0, eps=eps) - 0.5 < eps

    def test_normal_cdf_full(self):
        eps = 1e-10
        assert normal_cdf(1e6, eps=eps) - 1 < eps

    def test_normal_cdf_plus_one(self):
        eps = 1e-5
        assert normal_cdf(1, eps=eps) - 0.84134 < eps

    def test_normal_cdf_minus_one(self):
        eps = 1e-5
        assert normal_cdf(-1, eps=eps) - 0.15866 < eps


if __name__ == '__main__':
    unittest.main()