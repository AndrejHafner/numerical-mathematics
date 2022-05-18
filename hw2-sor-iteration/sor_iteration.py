import pickle

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd
import seaborn as sns
from tqdm import tqdm


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
        if isinstance(index, int) or isinstance(index, np.int64) or isinstance(index, np.int32):
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
        """
        Retrieve item from a certain index in the matrix.
        :param index: (row, column) tuple that represents the index of the value to return
        :return: Value at the index
        """
        self.__validate_index(index)
        i, j = index
        v_col_idx = np.where(self.I[i, :] == j)[0]
        return 0 if len(v_col_idx) == 0 else self.V[i, v_col_idx]

    def __setitem__(self, index, value):
        """
        Set the item at a certain index in the matrix.
        :param index: (row, column) tuple that represents the index at which to set the value
        :param value: value to be set
        """
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
        """
        Implementation of sparse matrix multiplication by a vector (from the right side).

        Example:
            A = SparseMatrix.from_np(np.diag(np.arange(2))
            b = np.arange(2)
            result = A @ b

        :param other: Vector which will be multiplied.
        :return: Multiplication result.
        """
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
        """
        Construct a SparseMatrix from a NumPy matrix.
        :param np_arr: np.ndarray with two dimensions
        :return: SparseMatrix instance
        """
        assert len(np_arr.shape) == 2, "NumPy array is not a matrix"
        assert np_arr.shape[0] == np_arr.shape[1], "NumPy array is not a square matrix"

        sparse_mat = SparseMatrix(len(np_arr))
        for i in range(len(np_arr)):
            for j in np.where(np_arr != 0)[0]:
                sparse_mat[i, j] = np_arr[i, j]

        return sparse_mat

    def to_np(self):
        """
        Convert this SparseMatrix to a np.ndarray.
        :return: NumPy ndarray
        """
        np_arr = np.zeros((len(self), len(self)))
        for i in range(len(self)):
            col_indices = np.where(self.I[i, :] >= 0)[0]
            col_values = self.V[i, col_indices]
            np_arr[i, self.I[i, col_indices]] = col_values

        return np_arr


def sor(A, b, x0, omega, tol=1e-10):
    """
    Implementation of SOR method for solving a linear system of equations where A is a sparse matrix.

    Example:
        A = SparseMatrix.from_np(np.diag(np.arange(5)**2))
        b = (np.arange(5) - 3) ** 2
        x0 = np.random.random(5)
        omega = 0.8
        solution, iterations = sor(A, b, x0, omega)

    :param A: SparseMatrix instance. Matrix needs to be diagonally dominant in order to ensure convergence. Another critetion is that its spectral radius is less than 1.
    :param b: np.array vector
    :param x0: Initial approximation
    :param omega: SOR parameter omega
    :param tol: Stopping criterion. When the max norm of Ax - b drops below this value we stop the iterations.
    :return: (solution, iterations) tuple where solution is the x from Ax = b system and iterations is the number of iterations it took for the method to converge
    """

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

def random_graph(n, p, seed=None):
    """
    Generate a Erdos-Renyi random graph.
    :param n: Number of nodes
    :param p: Probability of an edge forming between two nodes
    :param seed: Random seed
    :return: (networkx.Graph, dict) tuple with a NetworkX graph and an adjancency list where keys are nodes and values are lists of neighbours
    """
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    return G, {node: list(neighs.keys()) for node, neighs in G.adjacency()}

def plot_2d_graph_embedding(x, y, adj_list):
    """
    Plot an embedding of a graph in 2D.
    :param x: Node positions on the x-axis.
    :param y: Node positions on the y-axis.
    :param adj_list: Adjacency lists of a graph
    """
    node_positions = {node: (x[node], y[node]) for node in range(len(adj_list))}

    plt.figure()

    for node in node_positions.keys():
        plt.scatter(*node_positions[node], s=200, color="black", zorder = 5, facecolor="white")

    for node in node_positions.keys():
        for neigh in adj_list[node]:
            x1, y1 = node_positions[node]
            x2, y2 = node_positions[neigh]
            plt.plot([x1, x2], [y1, y2], marker="o", color="g",
                     markeredgecolor="black", markerfacecolor="white", markersize=0)
    plt.title("Erdos-Renyi graph embedding")
    plt.show()

def create_2d_graph_embedding(adj_list, omega):
    """
    Embed a graph into a plane using a force directed drawing method. The linear system is solved using the SOR method.
    :param adj_list: Adjacency list of a graph.
    :param omega: Omega value for SOR method.
    :return: (solution, iterations) tuple where solution is the x from Ax = b system and iterations is the number of iterations it took for the SOR method to converge
    """
    A = SparseMatrix(2 * len(adj_list))
    for node in adj_list.keys():
        for neigh in adj_list[node]:
            A[2 * node, 2 * neigh] = 1
            A[2 * node + 1, 2 * neigh + 1] = 1
        A[2 * node, 2 * node] = -len(adj_list[node])
        A[2 * node + 1, 2 * node + 1] = -len(adj_list[node])

    b = np.zeros(2 * len(adj_list))
    x0 = np.random.random(2 * len(adj_list))

    # We fix the nodes on a circle, equally spaced around
    fixed_nodes = np.random.choice(range(len(adj_list)), 5)
    r = 50

    A = A.to_np()
    for idx, node in enumerate(fixed_nodes):
        for j in range(len(A)):
            A[2 * node, j] = 0
            A[2 * node + 1, j] = 0
        A[2 * node, 2 * node] = 1
        A[2 * node + 1, 2 * node + 1] = 1
        b[2 * node] = r * np.cos((idx / len(fixed_nodes) * 360 * (np.pi / 180)))
        b[2 * node + 1] = r * np.sin((idx / len(fixed_nodes) * 360 * (np.pi / 180)))

    return sor(SparseMatrix.from_np(A), b, x0, omega, tol=1e-6)

def graph_plot_example():
    """
    Plot an example of a graph embedding.
    """
    n = 20
    omega = 0.6

    _, adj_list = random_graph(n, 0.2)

    solution, it = create_2d_graph_embedding(adj_list, omega)

    x = solution[0::2]
    y = solution[1::2]
    plot_2d_graph_embedding(x, y, adj_list)


def check_omega_param_effect():
    """
    Check the effect of omega parameter on the convergence speed of SOR method.
    """
    n = 50
    p = 0.5
    repetitions = 50
    omegas = np.array(range(5, 105, 5)) / 100

    iterations = np.zeros((repetitions, len(omegas)))

    for rep in tqdm(range(repetitions)):
        for col, omega in enumerate(omegas):
            _, adj_list = random_graph(n, p)
            _, it = create_2d_graph_embedding(adj_list, omega)
            iterations[rep, col] = it

    with open(f"iterations_n_{n}_reps_{repetitions}.pickle", "wb") as f:
        pickle.dump(iterations, f)

def plot_omegas():
    """
    Plot a graph of effect of omega parameter on the convergence speed of SOR method.
    """
    n = 50
    repetitions = 50

    with open(f"iterations_n_{n}_reps_{repetitions}.pickle", "rb") as f:
        iterations = pickle.load(f)

    omegas = np.array(range(5, 105, 5)) / 100
    df = pd.DataFrame()
    df["omegas"] = np.concatenate([np.repeat(omegas[i], 50) for i in range(20)])
    df["iterations"] = iterations.T.flatten()
    sns.lineplot(data=df, x="omegas", y="iterations", ci=95)
    plt.xlabel("Omega")
    plt.ylabel("Iterations until convergence")
    plt.title("SOR convergence speed with 95% confidence interval")
    plt.show()


def main():
    graph_plot_example()
    # check_omega_param_effect()
    # plot_omegas()

if __name__ == '__main__':
    main()
