from matplotlib import animation
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from screeninfo import get_monitors
import time

matplotlib.use('TkAgg')

for m in get_monitors():
    height = m.height
    width = m.width
wait = 5
mode = input("Please give either 1 or 2: ")

fig = plt.figure()
ax = plt.axes(xlim=(-width / 2, width / 2), ylim=(-height / 2, height / 2))
line, = ax.plot([], [], lw=5)

x_data, y_data = [], []


def init():
    line.set_data([], [])
    return line,


def animate_spiral(i):
    t = 0.05 * i
    x = t * t * np.sin(t)
    y = t * t * np.cos(t)
    x_data.append(x)
    y_data.append(y)
    if not (len(x_data) > 0 and (x_data[len(x_data) - 1] > width / 2 or x_data[len(x_data) - 1] < -width / 2) or
            len(y_data) > 0 and (y_data[len(y_data) - 1] > height / 2 or y_data[len(y_data) - 1] < -height / 2)):
        line.set_data(x_data, y_data)
    else:
        time.sleep(wait)
        exit()

    return line,


def animate_rect(i):


    return line,


if mode == "1":
    anim = animation.FuncAnimation(fig, animate_spiral, init_func=init, frames=500, interval=20, blit=True)
if mode == "2":
    anim = animation.FuncAnimation(fig, animate_rect, init_func=init, frames=500, interval=20, blit=True)

manager = plt.get_current_fig_manager()
manager.window.state('zoomed')
plt.show()
