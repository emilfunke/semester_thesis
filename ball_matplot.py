from matplotlib import animation
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from screeninfo import get_monitors
import time
from datetime import datetime
import csv
from itertools import zip_longest

matplotlib.use('TkAgg')

for m in get_monitors():
    height = m.height
    width = m.width
wait = 5
mode = input("Please give either 1, 2 or 3: ")

fig = plt.figure()
ax = plt.axes(xlim=(-width / 2, width / 2), ylim=(-height / 2, height / 2))
point, = ax.plot(0, 1, marker="o")
line, = ax.plot([], [], lw=5)

x_data, y_data, time_data = [], [], []
r = 400  # radius of circle


def init():
    line.set_data([], [])
    return line,


def animate_spiral(i):
    t = 0.03 * i
    x = t * t * np.sin(t)
    y = t * t * np.cos(t)
    now = datetime.now()
    x_data.append(x)
    y_data.append(y)
    time_data.append(now)
    if not (len(x_data) > 0 and (x_data[len(x_data) - 1] > width / 2 or x_data[len(x_data) - 1] < -width / 2) or
            len(y_data) > 0 and (y_data[len(y_data) - 1] > height / 2 or y_data[len(y_data) - 1] < -height / 2)):
        line.set_data(x_data, y_data)
    else:
        time.sleep(wait)
        plt.close(fig)
    return line,


def animate_rect(i):


    return point,


def circle(phi):
    return np.array([r * np.cos(phi), r * np.sin(phi)])


def animate_circle(phi):
    x, y = circle(phi)
    point.set_data([x], [y])
    return point,


if mode == "1":
    anim = animation.FuncAnimation(fig, animate_spiral, init_func=init, frames=850, interval=0, blit=True, repeat=False)
elif mode == "2":
    anim = animation.FuncAnimation(fig, animate_rect, init_func=init, frames=100, interval=0, blit=True, repeat=False)
elif mode == "3":
    anim = animation.FuncAnimation(fig, animate_circle, interval=0, blit=True, repeat=False,
                                   frames=np.linspace(0, 4 * np.pi, 1440, endpoint=False))
else:
    print("fuck you start new")


manager = plt.get_current_fig_manager()
manager.window.state('zoomed')
plt.show()

if mode == "1":
    temp = [x_data, y_data, time_data]
    exp_data = zip_longest(*temp, fillvalue='')
    with open('spiral.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(("x_data", "y_data", "time"))
        wr.writerows(exp_data)
    myfile.close()

if mode == "2":
    temp = [x_data, y_data, time_data]
    exp_data = zip_longest(*temp, fillvalue='')
    with open('rect.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(("x_data", "y_data", "time"))
        wr.writerows(exp_data)
    myfile.close()

if mode == "3":
    temp = [x_data, y_data, time_data]
    exp_data = zip_longest(*temp, fillvalue='')
    with open('circle.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(("x_data", "y_data", "time"))
        wr.writerows(exp_data)
    myfile.close()
