import matplotlib.pyplot as plt
import numpy as np

from animate import animate_pendulum, create_gif

G = 9.80665

def rk4(f, y0, t):
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

def pendulum(l, t, theta0, dtheta0, n):
    pend_system_fun = lambda t_, y: np.array([y[1], -G/l * np.sin(y[0])])
    t_eval = np.linspace(0, t, n)
    sol = rk4(pend_system_fun, np.array([theta0, dtheta0]), t_eval)

    return sol[:, 0], sol[:, 1]

def plot_period_dependence_on_energy():
    m = 1 # Assume a mass of 1kg on the pendulum
    l = 1 # Length of pendulum is 1m
    I = m * l**2 # Moment of inertia of the pendulum
    t = 5
    n = 10000
    h = t / n
    init_angle = 0 # All the energy is kinetic energy
    init_velocities = np.linspace(0.1, 5.4, 54) # Velocities in radians
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


def main():
    plot_period_dependence_on_energy()
    # pendulum(1, 3, 0, 1, 150)

if __name__ == '__main__':
    main()