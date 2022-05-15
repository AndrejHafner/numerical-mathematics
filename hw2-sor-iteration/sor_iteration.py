import numpy as np

class IndexException(Exception):
    pass

class IndexOutOfBounds(IndexException):
    pass

class IndexDatatypeNotImplemented(IndexException):
    pass

class IndexTupleLengthMismatch(IndexException):
    pass

class SparseMatrix:

    def __init__(self, n):
        self.n = n
        self.V = np.zeros((self.n, 1))
        self.I = -np.ones((self.n, 1), dtype=int)

    def __validate_index(self, index):
        """
        Validate the index passed in __set_item__ and __get_item__, raise exception otherwise.
        :param index: Index value
        """
        if isinstance(index, int) or isinstance(index, np.int64):
            if index < 0 or index >= len(self):
                raise IndexOutOfBounds(f"Integer index with value {index} is out of bounds on matrix with shape {(self.n, self.n)}.")
        elif isinstance(index, tuple):
            if len(index) != 2:
                raise IndexTupleLengthMismatch("Length of tuple passed as an index doesn't match the dimensionality of the matrix")
            for el in index:
                self.__validate_index(el)
        else:
            raise IndexDatatypeNotImplemented(f"Index of data type {type(index)} is not supported.")

    def __getitem__(self, index):
        self.__validate_index(index)
        i, j = index
        v_col_idx = np.where(self.I[i, :] == j)[0]
        return 0 if len(v_col_idx) == 0 else self.V[i, v_col_idx]


    def __setitem__(self, index, value):
        self.__validate_index(index)
        i, j = index
        v_col_idx = np.where(self.I[i, :] == j)

        if len(v_col_idx[0]) == 0:
            empty_indices = np.where(self.I[i, :] == -1)[0]
            if len(empty_indices) > 0:
                self.V[i, empty_indices[0]] = value
                self.I[i, empty_indices[0]] = int(j)
            else:
                self.V = np.hstack([self.V, -np.ones((len(self), 1), dtype=int)])
                self.I = np.hstack([self.I, -np.ones((len(self), 1), dtype=int)])
                self.V[i, -1] = value
                self.I[i, -1] = int(j)
        else: # Overwrite an existing value
            self.V[i, v_col_idx] = value

    def __matmul__(self, other):
        assert isinstance(other, np.ndarray), "Item on the right is not a np.ndarray"
        assert len(other.shape) <= 2, "Item on the right is not a vector"

        if len(other.shape) == 2:
            assert min(other.shape) == 1, "Item on the right is not a vector"
            other = other.squeeze() # We want to work with 1D vectors

        assert len(other) == len(self), "Matrix and vector dimensions don't match"

        result = np.zeros(len(self))

        for i in range(len(self)):
            result[i] = self.V[i, :] @ other[np.where(self.I[i, :] >= 0)[0]]

        return result

    def __len__(self):
        return self.n

    @staticmethod
    def from_np(np_arr):
        assert len(np_arr.shape) == 2, "NumPy array is not a matrix"
        assert np_arr.shape[0] == np_arr.shape[1], "NumPy array is not a square matrix"

        sparse_mat = SparseMatrix(len(np_arr))
        for i in range(len(np_arr)):
            for j in np.where(np_arr != 0)[0]:
                sparse_mat[i, j] = np_arr[i, j]

        return sparse_mat

    def to_np(self):
        np_arr = np.zeros((len(self), len(self)))
        for i in range(len(self)):
            col_indices = np.where(self.I[i, :] >= 0)[0]
            col_values = self.V[i, col_indices]
            np_arr[i, self.I[i, col_indices]] = col_values

        return np_arr

def gen_rand_band_matrix(n, band_width):
    mat = np.zeros((n, n))
    side_diags = (band_width - 1) // 2
    for i in range(n):
        for j in range(max(0, i - side_diags), min(n, i + side_diags + 1)):
            mat[i][j] = np.random.randint(0, 20)

    return mat

def sor_np(A, b, x0, omega, tol=1e-10):
    x_k = x0.copy()
    iteration = 0

    while np.linalg.norm(A @ x_k  - b, ord=np.inf) >= tol:
        iteration += 1

        for i in range(len(x_k)):
            x_k[i] = (1 - omega) * x_k[i] +  (omega / A[i, i]) * (b[i] - A[i, :i] @ x_k[:i] - A[i, i+1:] @ x_k[i+1:])

        norm = np.linalg.norm(A @ x_k  - b, ord=np.inf)
        print(f"NP Iteration: {iteration}, max. norm: {norm}")

    return iteration, x_k

def sor(A, b, x0, omega, tol=1e-10):

    assert isinstance(A, SparseMatrix), "A is not an instance of SparseMatrix"

    x_k = x0.copy()
    it = 0

    while np.linalg.norm(A @ x_k - b, ord=np.inf) >= tol:
        it += 1

        for i in range(len(x_k)):
            s1_A_row = np.zeros(i)
            s1_A_I_indices = A.I[i, :] < i
            s1_A_I_col_indices = A.I[i, s1_A_I_indices]
            s1_A_row[s1_A_I_col_indices] = A.V[i, s1_A_I_indices]
            s1 = s1_A_row @ x_k[:i]

            s2_A_row = np.zeros(len(x_k) - i - 1)
            s2_A_I_indices = A.I[i, :] > i
            s2_A_I_col_indices = A.I[i, s2_A_I_indices]
            s2_A_row[s2_A_I_col_indices - i - 1] = A.V[i, s2_A_I_indices]
            s2 = s2_A_row @ x_k[i+1:]

            x_k[i] = (1 - omega) * x_k[i] +  (omega / A[i, i]) * (b[i] - s1 - s2)

    return x_k, it


def main():
    pass

if __name__ == '__main__':
    main()
