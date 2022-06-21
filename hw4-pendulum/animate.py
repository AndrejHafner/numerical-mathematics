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
