import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def animate_pendulum(t, t_eval, l, angle, speed):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax2.axis([0, t, min(min(angle), min(speed)) - 2, max(max(angle), max(speed)) + 2])

    for i in range(len(angle)):
        ax1.plot([0, l * np.sin(angle[i])], [0, -l * np.cos(angle[i])], color="black")
        ax1.plot(l * np.sin(angle[i]), -l * np.cos(angle[i]), ".", markersize=30)
        ax2.plot(t_eval[:i], angle[:i], '-', color='b', label="Angle")
        ax2.plot(t_eval[:i], speed[:i], '-', color='orange', label="Speed")
        if(i == 0):
            plt.legend()

        plt.draw()
        plt.pause((t_eval[1] - t_eval[0]))
        ax1.clear()
        ax1.axis([-1.5*l, 1.5*l, -1.5*l, 1.5*l])



def animate_math_and_harm_pend(t, t_eval, l, angle_math, speed_math, angle_harm, speed_harm, filename="harm_math_pendulum_comparison"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    def animate(i):
        ax1.clear()
        ax2.clear()

        ax2.axis([0, t, min(min(angle_math), min(speed_math)) - 2, max(max(angle_math), max(speed_math)) + 2])
        ax1.axis([-1.5*l, 1.5*l, -1.5*l, 1.5*l])

        l1, = ax1.plot([0, l * np.sin(angle_math[i])], [0, -l * np.cos(angle_math[i])], color="black")
        l2, = ax1.plot([0, l * np.sin(angle_harm[i])], [0, -l * np.cos(angle_harm[i])], color="black")
        l3, = ax1.plot(l * np.sin(angle_math[i]), -l * np.cos(angle_math[i]), ".", color='cornflowerblue', markersize=30)
        l4, = ax1.plot(l * np.sin(angle_harm[i]), -l * np.cos(angle_harm[i]), "^", color="orange", markersize=15)

        l4, = ax2.plot(t_eval[:i], angle_math[:i], '-', color='cornflowerblue', label="Angle - Math")
        l5, = ax2.plot(t_eval[:i], speed_math[:i], '--', color='cornflowerblue', label="Speed - Math")
        l6, = ax2.plot(t_eval[:i], angle_harm[:i], '-', color="orange", label="Angle - Harmonic")
        l7, = ax2.plot(t_eval[:i], speed_harm[:i], '--', color="orange", label="Speed - Harmonic")
        plt.legend(loc="upper left")
        return l1, l2, l3, l4, l5, l6, l7,

    interval = round((t_eval[1] - t_eval[0]) * 1000)
    fps = round(1000 / interval)

    ani = FuncAnimation(fig, animate, interval=interval, blit=True, repeat=True, frames=len(t_eval))
    ani.save(f"{filename}.gif", dpi=300, writer=PillowWriter(fps=fps))

def create_gif(t, t_eval, l, angle, speed, filename="pendulum"):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    def animate(i):
        ax1.clear()
        ax2.clear()
        ax1.axis([-1.5 * l, 1.5 * l, -1.5 * l, 1.5 * l])
        ax2.axis([0, t, min(min(angle), min(speed)) - 2, max(max(angle), max(speed)) + 2])
        l1, = ax1.plot([0, l * np.sin(angle[i])], [0, -l * np.cos(angle[i])], color="black")
        l2, = ax1.plot(l * np.sin(angle[i]), -l * np.cos(angle[i]), ".", markersize=30)
        l3, = ax2.plot(t_eval[:i], angle[:i], '-', color='b', label="Angle")
        l4, = ax2.plot(t_eval[:i], speed[:i], '-', color='orange', label="Speed")
        ax2.legend()

        return l1, l2, l3, l4

    interval = round((t_eval[1] - t_eval[0]) * 1000)
    fps  = round(1000 / interval)

    ani = FuncAnimation(fig, animate, interval=interval, blit=True, repeat=True, frames=len(t_eval))
    ani.save(f"{filename}.gif", dpi=300, writer=PillowWriter(fps=fps))
