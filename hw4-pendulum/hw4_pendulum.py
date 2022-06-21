import matplotlib.pyplot as plt
import numpy as np

from animate import animate_pendulum, create_gif, animate_math_and_harm_pend

G = 9.80665 # Earths gravity acceleration constant


def rk4(f, y0, t):
    """
    Runge-Kutta method of order 4 for solving a system of ordinary differential equations.

    Example:
        import numpy as np
        diff_func = lambda x, y: np.array([y[1], -G/l * np.sin(y[0])])
        solution = rk4(diff_func, np.array([0, 1]), 5)

    :param f: Differential function we want to solve
    :param y0: Initial conditions
    :param t: A list of time values for which we want to solve the differential equation.
    :return: Solution of differential equation for all time values. Has the shape of (len(t), len(y0)).
    """
    y = np.zeros((len(t), len(y0)))
    y[0, :] = y0
    h = t[1] - t[0]

    for i in range(0, len(t) - 1):
        k1 = h * f(t[i], y[i])
        k2 = h * f(t[i] + h/2, y[i] + k1/2)
        k3 = h * f(t[i] + h/2, y[i] + k2/2)
        k4 = h * f(t[i] + h, y[i] + k3)
        y[i+1, :] = y[i] + (k1 + 2*k2 + 2*k3 + k4)/6

    return y


def pendulum(l, t, theta0, dtheta0, n, harmonic=False):
    """
    Simulate the movement of a simple pendulum.

    Example:
        angles, velocities = pendulum(1.5, 5, 0, 2.5, 150)

    :param l: Length of the pendulum
    :param t: Defines the interval in which we will solve the differential equation - [0, t]
    :param theta0: Initial angle
    :param dtheta0: Initial angular velocity
    :param n: Number of subintervals we split the interval [0, t] into.
    :param harmonic: Defaults to false. If false, we use the differential equation for a simple pendulum and for a harmonic oscillator otherwise. Used only for internal testing.
    :return: Tuple of angles and angular velocities of the pendulum at the different time steps.
    """
    if harmonic:
        pend_system_fun = lambda t_, y: np.array([y[1], -G/l * y[0]])
    else:
        pend_system_fun = lambda t_, y: np.array([y[1], -G/l * np.sin(y[0])])
    t_eval = np.linspace(0, t, n)
    sol = rk4(pend_system_fun, np.array([theta0, dtheta0]), t_eval)
    return sol[:, 0], sol[:, 1]


def nihalo(l, t, theta0, dtheta0, n):
    """
    Simulate the movement of the pendulum - same as the function pendulum(..), but we return the angle at the time t.

    Example:
        angle_at_t = nihalo(1.5, 5, 0, 2.5, 150)

    :param l: Length of the pendulum
    :param t: Defines the interval in which we will solve the differential equation - [0, t]
    :param theta0: Initial angle
    :param dtheta0: Initial angular velocity
    :param n: Number of subintervals we split the interval [0, t] into.
    :param harmonic: Defaults to false. If false, we use the differential equation for a simple pendulum and for a harmonic oscillator otherwise. Used only for internal testing.
    :return: Angle at time t.
    """
    angles, _ = pendulum(l, t, theta0, dtheta0, n)
    return angles[-1, 0]


def plot_period_dependence_on_energy():
    """
    Analyze how the period of the pendulum changes with respect to the initial angular velocity.
    """
    m = 1 # Assume a mass of 1kg on the pendulum
    l = 1 # Length of pendulum is 1m
    I = m * l**2 # Moment of inertia of the pendulum
    t = 5
    n = 10000
    h = t / n
    init_angle = 0 # All the energy is kinetic energy
    init_velocities = np.linspace(0.1, 5.45, 55) # Velocities in radians
    rot_energy = 0.5 * I * init_velocities**2

    periods = []
    for i in range(len(init_velocities)):
        angles, speed = pendulum(l, t, init_angle, init_velocities[i], n)
        speed_rounded = np.round(np.abs(speed), decimals=2)
        zero_speed_indices = np.argwhere(speed_rounded == 0)
        selected_indices = []
        mean_indices = []
        for j in range(1, len(zero_speed_indices)):
            if abs(zero_speed_indices[j-1] - zero_speed_indices[j]) > 5:
                selected_indices.append(np.mean(mean_indices))
                mean_indices.clear()
            else:
                mean_indices.append(zero_speed_indices[j][0])
        periods.append((selected_indices[2] - selected_indices[0]) * h)

    plt.plot(rot_energy, periods)
    plt.xlabel("Rotational energy")
    plt.ylabel("Period")
    plt.show()


def compare_math_harmonic_pendulum():
    """
    Compare the system of a simple pendulum and harmonic oscillator at different starting conditions.
    """
    l = 2
    t = 10
    init_angle = 0
    init_speed = 0.3
    n = 300
    t_eval = np.linspace(0, t, n)

    math_angle, math_speed = pendulum(l, t, init_angle, init_speed, n)
    harm_angle, harm_speed = pendulum(l, t, init_angle, init_speed, n, harmonic=True)
    animate_math_and_harm_pend(t, t_eval, l, math_angle, math_speed, harm_angle, harm_speed, filename="harm_math_pendulum_comparison_small_angle")


def main():
    # compare_math_harmonic_pendulum()
    plot_period_dependence_on_energy()
    # pendulum(1.5, 5, 0, 2.5, 150)

if __name__ == '__main__':
    main()