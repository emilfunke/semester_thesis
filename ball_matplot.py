from matplotlib import animation
from matplotlib import pyplot as plt
import numpy as np
from screeninfo import get_monitors

for m in get_monitors():
    height = m.height
    width = m.width

fig = plt.figure()
ax = plt.axes(xlim=(0, width), ylim=(0, height))
line, = ax.plot([], [], lw=2)

x_data, y_data = [], []


def init():
    line.set_data([], [])
    return line,


def animate(i):
    x_data.append(i)
    y_data.append(i)

    line.set_data(x_data, y_data)
    return line,


anim = animation.FuncAnimation(fig, animate, init_func=init, frames=100)
