import time
import timeit
from functools import lru_cache

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm

cdf_int = lambda t: np.exp(-(t ** 2 / 2))

class Timer:
    def __init__(self, output):
        self.output = output

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"{self.output} --> {round(time.time() - self.start_time, 6)}s")

def simpson(f, a, fa, b, fb):
    h = a + (b - a) / 2
    fh = f(h)
    int_val = np.abs(b - a) / 6 * (fa + 4 * fh + fb)
    return h, fh, int_val

def _adaptive_simpson(f, a, fa, b, fb, h, fh, int_val, eps):
    lh, flh, l_int_val = simpson(f, a, fa, h, fh)
    rh, rlh, r_int_val = simpson(f, h, fh, b, fb)
    err = l_int_val + r_int_val - int_val
    if np.abs(err) <= 15 * eps:
        return l_int_val + r_int_val + err / 15
    return _adaptive_simpson(f, a, fa, h, fh, lh, flh, l_int_val, eps / 2) + _adaptive_simpson(f, h, fh, b, fb, rh, rlh, r_int_val, eps / 2)

def adaptive_simpson(f, a, b, eps=1e-10):
    fa, fb = f(a), f(b)

    h, fh, int_val = simpson(f, a, fa, b, fb)
    # Adaptive step - recursively calculate the step size in order to reduce the error below eps
    return _adaptive_simpson(f, a, fa, b, fb, h, fh, int_val, eps)

@lru_cache
def _normal_cdf_lb(eps):
    lb_candidates = -np.logspace(0, np.log(np.iinfo(np.int32).max), 1000)
    f = lambda t: (1 / np.sqrt(2 * np.pi)) * cdf_int(t)
    for i in range(1, len(lb_candidates)):
        if np.abs(f(lb_candidates[i] - f(lb_candidates[i-1]))) < eps:
            return lb_candidates[i]

def normal_cdf(x, eps=1e-15):
    lower_bound = _normal_cdf_lb(eps)
    return (1 / np.sqrt(2 * np.pi)) * adaptive_simpson(cdf_int, lower_bound, x, eps=eps)

def plot_eps_performance():
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