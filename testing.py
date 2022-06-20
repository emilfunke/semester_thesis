import os
import pandas as pd


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def get_x_y(mode):
    if mode == 1:
        circle1 = pd.read_csv("csv/circle1.csv")
        circle_x = circle1['x_data']
        circle_y = circle1['y_data']
        x_arr = circle_x.to_numpy()
        y_arr = circle_y.to_numpy()
    if mode == 2:
        rect1 = pd.read_csv("csv/rect1.csv")
        rect_x = rect1['x_data']
        rect_y = rect1['y_data']
        x_arr = rect_x.to_numpy()
        y_arr = rect_y.to_numpy()
    return x_arr, y_arr


def get_paths_1_by_1(mode, value):
    paths = []
    if mode == 1:
        path = "camera_output/circle" + str(value) + "/"
        for name in os.listdir(path):
            if name.endswith("jpg"):
                paths.append(path + name)
    if mode == 2:
        path = "camera_output/rect" + str(value) + "/"
        for name in os.listdir(path):
            if name.endswith("jpg"):
                paths.append(path + name)
    return paths


circle1 = get_paths_1_by_1(1, 1)
x_split, y_split = get_x_y(1)

x_split = list(split(x_split, len(circle1)))
x = []
for i in range(len(x_split)):
    tot = 0
    for j in range(len(x_split[i])):
        tot += x_split[i][j]
    x.append(int(tot / len(x_split[i])))

print(x, len(x))
