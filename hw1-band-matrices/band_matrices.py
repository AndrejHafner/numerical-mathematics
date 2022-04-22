from functools import partial
from typing import Tuple, Union

import numpy as np
import operator

class IndexException(Exception):
    pass

class IndexOutOfBounds(IndexException):
    pass

class IndexDatatypeNotImplemented(IndexException):
    pass

class IndexTupleLengthMismatch(IndexException):
    pass

class IndexItemSetOutOfBand(IndexException):
    pass





class BandMatrix:
    """
    TODO: Add docs
    """

    def __init__(self, n: int, band_width: int = 1) -> None:

        assert band_width % 2 == 1 and band_width >= 1, "Band width is not an odd number or is not positive."
        assert (band_width - 1) // 2 < n, f"Band width is too large, should be smaller than the size of the matrix: {n}."

        self.__shape = (n, n)
        self.__band_width = band_width # needs to be an odd number
        self.__n_side_diags = (band_width - 1) // 2 # number of diagonals above the main diagonal or below the main diagonal
        self.main_diag = np.zeros(n) # main diagonal
        self.lower_diags = np.zeros((self.__n_side_diags, n - 1)) # diagonals below the main
        self.upper_diags = np.zeros((self.__n_side_diags, n - 1)) # diagonals above the main

    @property
    def shape(self) -> Tuple[int, int]:
        return self.__shape

    @property
    def band_width(self) -> int:
        return self.__band_width

    def __validate_index(self, index):
        if isinstance(index, int):
            if index < 0 or index >= len(self):
                raise IndexOutOfBounds(f"Integer index with value {index} is out of bounds on matrix with shape {self.__shape}.")
        elif isinstance(index, tuple):
            if len(index) != len(self.__shape):
                raise IndexTupleLengthMismatch("Length of tuple passed as an index doen't match the dimensionality of the matrix")
            for el in index:
                self.__validate_index(el)
        else:
            raise IndexDatatypeNotImplemented(f"Index of data type {type(index)} is not supported.")

    def __is_index_in_band(self, i, j):
        return abs(i - j) <= self.__n_side_diags

    def __get_map_index_to_band(self, i, j):
        if i == j:
            return self.main_diag[i]
        elif i > j:
            nth_diag = i - j - 1
            return self.lower_diags[nth_diag][j]
        elif i < j:
            nth_diag = j - i - 1
            return self.upper_diags[nth_diag][i]

    def __set_map_index_to_band(self, i, j, el):
        if i == j:
            self.main_diag[i] = el
        elif i > j:
            nth_diag = i - j - 1
            self.lower_diags[nth_diag][j] = el
        elif i < j:
            nth_diag = j - i - 1
            self.upper_diags[nth_diag][i] = el

    def __getitem__(self, key: Tuple[int, int]):
        self.__validate_index(key)
        return self.__get_map_index_to_band(*key) if self.__is_index_in_band(*key) else 0

    def __setitem__(self, key: Tuple[int, int], value: Union[int, float, bool]):
        self.__validate_index(key)
        if self.__is_index_in_band(*key):
            return self.__set_map_index_to_band(*key, value)
        else:
            raise IndexItemSetOutOfBand(f"Item tried to be set outside the band of the matrix, key: f{key}.")

    def __add__(self, other):
        assert self.shape == other.shape, f"Band matrix shape mismatch: {self.shape} != {other.shape}"
        assert self.band_width == other.band_width, f"Band matrix band width mismatch: {self.band_width} != {other.band_width}"
        return BandMatrix.from_diags(self.main_diag + other.main_diag, self.lower_diags + other.lower_diags, self.upper_diags + other.upper_diags)

    def __sub__(self, other):
        assert self.shape == other.shape, f"Band matrix shape mismatch: {self.shape} != {other.shape}"
        assert self.band_width == other.band_width, f"Band matrix band width mismatch: {self.band_width} != {other.band_width}"
        return BandMatrix.from_diags(self.main_diag - other.main_diag, self.lower_diags - other.lower_diags, self.upper_diags - other.upper_diags)

    def __mul_div(self, other, operation=None):
        assert any([isinstance(other, _type) for _type in [self.__class__, float, int]]), "cannot multiply with types other than BandMatrix, int or float"

        if isinstance(other, self.__class__):
            assert self.shape == other.shape, f"Band matrix shape mismatch: {self.shape} != {other.shape}"
            assert self.band_width == other.band_width, f"Band matrix band width mismatch: {self.band_width} != {other.band_width}"
            return BandMatrix.from_diags(operation(self.main_diag, other.main_diag),
                                         operation(self.lower_diags, other.lower_diags),
                                         operation(self.upper_diags, other.upper_diags))
        elif isinstance(other, int) or isinstance(other, float):
            main_diag = operation(self.main_diag, other)
            lower_diags = operation(self.lower_diags, other)
            upper_diags = operation(self.upper_diags, other)
            return BandMatrix.from_diags(main_diag, lower_diags, upper_diags)

    def __mul__(self, other):
        return self.__mul_div(other, operation=operator.mul)

    def __rmul__(self, other):
        return self.__mul_div(other, operation=operator.mul)

    def __truediv__(self, other):
        return self.__mul_div(other, operation=operator.truediv)

    def __rtruediv__(self, other):
        return self.__mul_div(other, operation=operator.truediv)

    def __floordiv__(self, other):
        return self.__mul_div(other, operation=operator.floordiv)

    def __rfloordiv__(self, other):
        return self.__mul_div(other, operation=operator.floordiv)

    def __matmul__(self, other):
        assert isinstance(other, np.ndarray), "Item on the right is not a np.ndarray"
        assert len(other.shape) <= 2, "Item on the right is not a vector"

        if len(other.shape) == 2:
            assert min(other.shape) == 1, "Item on the right is not a vector"
            other = other.squeeze() # We want to work with 1D vectors

        assert len(other) == self.shape[1], "Matrix and vector dimensions don't match"

        result = np.zeros_like(other)


        diagonals = np.zeros((self.__band_width, len(self)))
        diagonals[self.__n_side_diags, :] = self.main_diag
        diagonals[:self.__n_side_diags, :-1] = np.flip(self.upper_diags, axis=0)
        diagonals[self.__n_side_diags + 1:, 1:] = self.lower_diags


        for i in range(self.__n_side_diags, len(diagonals)):
            diagonals[i] = np.roll(diagonals[i], -(self.__n_side_diags - i + 1))

        diagonals = np.flip(diagonals, axis=0)

        for i in range(diagonals.shape[1]):
            if i < self.__n_side_diags:
                result[i] = diagonals[self.__n_side_diags - i:, i] @ other[:diagonals.shape[0] - (self.__n_side_diags - i)]
            elif i < diagonals.shape[1] - self.__n_side_diags:
                result[i] = diagonals[:, i].squeeze() @ other[i - self.__n_side_diags:i + diagonals.shape[0] - self.__n_side_diags]
            else:
                el1 = diagonals[:diagonals.shape[0] - (self.__n_side_diags - (diagonals.shape[1] - i) + 1), i].squeeze()
                el2 = other[i - (self.__n_side_diags):]
                result[i] = el1 @ el2

        np_mat = self.to_np_matrix()

        return result

        # # Multiply the first (n_side diags + 1) rows
        # result[0] = np.concatenate([np.array([self.main_diag[0]]), self.upper_diags[:, 0].squeeze()]) @ other[:(1 + self.__n_side_diags)]
        #
        # # Roll lower diagonals to allow for easier calculation
        # lower_diags_rolled = self.lower_diags.copy()
        # for i in range(1, self.__n_side_diags):
        #     lower_diags_rolled[i] = np.roll(lower_diags_rolled[i], -i)
        #
        # # Multiply the rows that don't include all lower diagonals
        # for i in range(1, self.__n_side_diags + 1):
        #     result[i] = np.concatenate([np.array(lower_diags_rolled[:i, i-1]), np.array([self.main_diag[0]]), self.upper_diags[:, i].squeeze()]) @ other[:(1 + self.__n_side_diags + i)]
        #


    def __len__(self):
        return self.__shape[0]

    def __repr__(self):
        return repr(self.to_np_matrix())

    def __str__(self):
        return str(self.to_np_matrix())

    def to_np_matrix(self) -> np.ndarray:
        mat = np.diag(self.main_diag)
        for i in range(1, self.__n_side_diags + 1):
            mat += np.diag(self.lower_diags[i - 1, :self.lower_diags.shape[1] - i + 1], k=-i)
            mat += np.diag(self.upper_diags[i - 1, :self.upper_diags.shape[1] - i + 1], k=i)
        return mat

    @staticmethod
    def from_diags(main_diag: np.array, lower_diags: np.ndarray, upper_diags):
        assert len(upper_diags) == len(lower_diags), f"Band width of upper and lower diagonals are not equal, {len(upper_diags)} != {len(upper_diags)}"
        assert (len(main_diag) - 1) == lower_diags.shape[1], f"Lower diagonals are too long, {lower_diags.shape[1]} != {len(main_diag) - 1}"
        assert (len(main_diag) - 1) == upper_diags.shape[1], f"Upper diagonals are too long, {upper_diags.shape[1]} != {len(main_diag) - 1}"

        band_mat = BandMatrix(len(main_diag), band_width=2 * len(lower_diags) + 1)
        band_mat.main_diag = main_diag
        band_mat.lower_diags = lower_diags
        band_mat.upper_diags = upper_diags

        return band_mat

    @staticmethod
    def from_np_matrix(mat: np.ndarray, band_width: int):
        assert len(mat.shape) == 2, "numpy.ndarray is not a matrix"
        assert mat.shape[0] == mat.shape[1], "numpy.ndarray is not a square matrix"
        assert band_width % 2 == 1, "band_width is not an odd number"
        assert band_width >= 0, "band_width is a negative number"

        # Check whether the given matrix is really a band matrix
        n_side_diags = (band_width - 1) // 2
        n = mat.shape[0]
        test_mat = np.ones(mat.shape)

        np.fill_diagonal(test_mat, 0)
        for i in range(1, n_side_diags + 1):
            test_mat -= np.diag(np.ones(n - i), k=-i)
            test_mat -= np.diag(np.ones(n - i), k=i)

        assert np.sum(mat * test_mat) == 0, "np.ndarray is not a band matrix with the given band_width"

        main_diag = np.diag(mat)
        lower_diags = np.zeros((n_side_diags, n - 1))
        upper_diags = np.zeros((n_side_diags, n - 1))
        for i in range(1, n_side_diags + 1):
            lower_diags[i - 1, :(n - i)] = np.diag(mat, k=-i)
            upper_diags[i - 1, :(n - i)] = np.diag(mat, k=i)

        return BandMatrix.from_diags(main_diag, lower_diags, upper_diags)



if __name__ == '__main__':
    n = 5
    a = BandMatrix(n, band_width=3)
    b = BandMatrix(n, band_width=3)

    a[1,1] = 5
    a[1,2] = 8

    b[1,1] = 3
    b[3,4] = -5


    print(a)
    print(b)
    print(a + b)

    mat = np.diag(np.ones(7))
    mat[3,2] = 1
    mat[4,3] = 5
    mat[5,6] = 10
    mat[0,2] = 2
    mat[1,2] = 8
    mat[6,6] = 8
    mat[4,6] = 8
    band_mat = BandMatrix.from_np_matrix(mat, 5)
    print(np.array_equal(mat, band_mat.to_np_matrix()))