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
wait = 3
mode = input("Please give either 1, 2 or 3: ")

fig = plt.figure()
ax = plt.axes(xlim=(-width / 2, width / 2), ylim=(-height / 2, height / 2))
point, = ax.plot(0, 1, marker="o")
line, = ax.plot([], [], lw=5)

x_data, y_data, time_data = [], [], []
r = 450  # radius of circle


def init():
    line.set_data([], [])
    point.set_data([], [])
    return line, point,


# https://towardsdatascience.com/animations-with-matplotlib-d96375c5442c
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


def rect():
    x_d, y_d = [], []
    for i in range(50):
        x_d.append(i * 10)
        y_d.append(i * 10)
    for i in range(30):
        x_d.append(i * 10 + 500)
        y_d.append(500)
    for i in range(100):
        x_d.append(800)
        y_d.append(500 - i * 10)
    for i in range(160):
        x_d.append(800 - i * 10)
        y_d.append(-500)
    for i in range(100):
        x_d.append(-800)
        y_d.append(-500 + i * 10)
    for i in range(130):
        x_d.append(-800 + i * 10)
        y_d.append(500)
    for i in range(100):
        x_d.append(500 - i * 10)
        y_d.append(500 - i * 10)
    for i in range(100):
        x_d.append(-500)
        y_d.append(-500 + i * 10)
    for i in range(100):
        x_d.append(-500 + i * 10)
        y_d.append(500)
    for i in range(100):
        x_d.append(500)
        y_d.append(500 - i * 10)
    for i in range(100):
        x_d.append(500 - i * 10)
        y_d.append(-500)
    return x_d, y_d


def animate_rect(i):
    if i < len(x_data):
        now = datetime.now()
        x = x_data[i]
        y = y_data[i]
        point.set_data([x], [y])
        time_data.append(now)
    else:
        time.sleep(wait)
        plt.close(fig)
    return point,


# https://stackoverflow.com/questions/51286455/how-can-i-animate-a-point-moving-around-the-circumference-of-a-circle
def circle(phi):
    return np.array([r * np.cos(phi), r * np.sin(phi)])


def animate_circle(phi):
    if phi < 4 * np.pi:
        now = datetime.now()
        x, y = circle(phi)
        point.set_data([x], [y])
        x_data.append(x)
        y_data.append(y)
        time_data.append(now)
    else:
        time.sleep(wait)
        plt.close(fig)
    return point,


if mode == "1":
    anim = animation.FuncAnimation(fig, animate_spiral, init_func=init, frames=850, interval=0, blit=True, repeat=False)
elif mode == "2":
    x_data, y_data = rect()
    anim = animation.FuncAnimation(fig, animate_rect, init_func=init, frames=len(x_data) + 1, interval=10, blit=True,
                                   repeat=False)
elif mode == "3":
    anim = animation.FuncAnimation(fig, animate_circle, interval=0, blit=True, repeat=False,
                                   frames=np.linspace(0, 4 * np.pi + 1, 1440, endpoint=False))
else:
    print("fuck you start new ")

manager = plt.get_current_fig_manager()
manager.window.state('zoomed')
plt.show()

# https://stackoverflow.com/questions/17704244/writing-python-lists-to-columns-in-csv
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
