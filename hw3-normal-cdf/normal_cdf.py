import timeit
from functools import lru_cache

import numpy as np
from matplotlib import pyplot as plt

# Part of Normal CDF that depends on t - relevant for integration
cdf_int = lambda t: np.exp(-(t ** 2 / 2))

def simpson(f, a, fa, b, fb):
    """
    Implementation of Simpson's rule for numerical integration, which works by quadratic interpolation on the interval and taking the integral of the parabola on the same interval.

    Example:
        import numpy as np
        sin_int = simpson(np.sin, 0, np.sin(0), np.pi, np.sin(np.pi)

    :param f: Function to integrate
    :param a: Lower bound
    :param fa: Functional value of function f at the lower bound
    :param b: Upper bound
    :param fb: Functional value of function f at the upper bound
    :return: Integration result
    """
    h = a + (b - a) / 2
    fh = f(h)
    int_val = np.abs(b - a) / 6 * (fa + 4 * fh + fb)
    return h, fh, int_val

def _adaptive_simpson(f, a, fa, b, fb, h, fh, int_val, eps):
    """
    Recursive implementation od the Adaptive Simpson method which recursively splits the integration subintervals (step h) in such a way to ensure that the error is below eps.

    This function is meant to be used only inside its wrapper named adaptive_simpson.

    :param f: Function to integrate
    :param a: Lower bound
    :param fa: Functional value of function f at the lower bound
    :param b: Upper bound
    :param fb: Functional value of function f at the upper bound
    :param h: Value of middle point between a and b
    :param fh: Functional value of function f at the middle point between a and b
    :param int_val: Value of the integral on the interval from a to b
    :param eps: Maximum allowed error in the integration value
    :return: Integration value
    """
    lh, flh, l_int_val = simpson(f, a, fa, h, fh)
    rh, rlh, r_int_val = simpson(f, h, fh, b, fb)
    err = l_int_val + r_int_val - int_val
    if np.abs(err) <= 15 * eps:
        return l_int_val + r_int_val + err / 15
    return _adaptive_simpson(f, a, fa, h, fh, lh, flh, l_int_val, eps / 2) + _adaptive_simpson(f, h, fh, b, fb, rh, rlh, r_int_val, eps / 2)

def adaptive_simpson(f, a, b, eps=1e-10):
    """
    Adaptive Simpson's method for numerical integration which calculates the value of an arbitrary function on the interval from a to b with an error below eps.

    Example:
        import numpy as np
        sin_int = adaptive_simpson(np.sin, 0, np.pi, eps=1e-6)

    :param f: Function to integrate (single variable input)
    :param a: Lower bound
    :param b: Upper bound
    :param eps: Maximal allowed error
    :return: Integration value
    """
    fa, fb = f(a), f(b)

    h, fh, int_val = simpson(f, a, fa, b, fb)
    # Adaptive step - recursively calculate the step size in order to reduce the error below eps
    return _adaptive_simpson(f, a, fa, b, fb, h, fh, int_val, eps)

@lru_cache
def _normal_cdf_lb(eps):
    """
    Function for calculation of lower bound for Normal CDF to ensure the maximal error eps. Function values are cached to ensure better performance.

    :param eps: Desired maximal error
    :return: Lower bound of Normal CDF
    """
    lb_candidates = -np.logspace(0, np.log(np.iinfo(np.int32).max), 1000)
    f = lambda t: (1 / np.sqrt(2 * np.pi)) * cdf_int(t)
    for i in range(1, len(lb_candidates)):
        if np.abs(f(lb_candidates[i] - f(lb_candidates[i-1]))) < eps:
            return lb_candidates[i]

def normal_cdf(x, eps=1e-10):
    """
    Calculation of cumulative density function for a normally distributed random variable X ~ N(0, 1).

    Example:
        import numpy as np
        p = normal_cdf(0)

    :param x:
    :param eps:
    :return:
    """
    lower_bound = _normal_cdf_lb(eps)
    return (1 / np.sqrt(2 * np.pi)) * adaptive_simpson(cdf_int, lower_bound, x, eps=eps)

def plot_eps_performance():
    """
    Plot the performance of normal_cdf at different values of epsilon.
    """
    eps_values = [np.power(10., -i) for i in range(1, 16)]
    times = np.zeros(len(eps_values))

    for idx, eps in enumerate(eps_values):
        times[idx] = timeit.timeit(lambda: normal_cdf(0, eps=eps), number=50)

    plt.plot(eps_values, times)
    plt.xscale("log")
    plt.yscale("log")
    plt.ylabel("time [s]")
    plt.xlabel("eps")
    plt.title("normal_cdf performance (50 iterations)")
    plt.show()

def main():
    plot_eps_performance()

if __name__ == '__main__':
    main()