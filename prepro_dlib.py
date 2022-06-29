import csv
import numpy as np
import pandas as pd
import os
import cv2
import math
import dlib


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


def get_paths(mode):
    # mode = 1 --> circle, mode = 2 --> rect
    paths = []
    if mode == 1:
        for i in range(1, 4, 1):
            path = "camera_output/circle" + str(i) + "/"
            for name in os.listdir(path):
                if name.endswith("jpg"):
                    paths.append(path + name)
    if mode == 2:
        for i in range(1, 7, 1):
            path = "camera_output/rect" + str(i) + "/"
            for name in os.listdir(path):
                if name.endswith("jpg"):
                    paths.append(path + name)
    return paths


def get_opt_flow_1_by_1(mode, value):
    paths = []
    if mode == 1:
        path = "camera_output/circle" + str(value) + "/"
        for name in os.listdir(path):
            if name.endswith("csv"):
                paths.append(path + name)
    if mode == 2:
        path = "camera_output/rect" + str(value) + "/"
        for name in os.listdir(path):
            if name.endswith("csv"):
                paths.append(path + name)
    return paths


def split(arr, n):
    k, m = divmod(len(arr), n)
    ret = []
    l = list((arr[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)))
    for i in range(n):
        tot = 0
        for j in range(len(l[i])):
            tot += l[i][j]
        ret.append(tot / len(l[i]))
    return ret


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


def mapping(paths, x, y):
    mapped = []
    for i in range(len(paths)):
        pair = [paths[i], x[i], y[i]]
        mapped.append(pair)
    return mapped


def get_face_roi(img_path):
    img = cv2.imread(img_path)
    p = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    detect = detector(img, 1)
    predictor = dlib.shape_predictor(p)
    shape = predictor(img, detect[0])

    # left eye
    x1_l, x2_l = int(shape.part(36).x * 0.9), int(shape.part(39).x * 1.1)
    y1_l, y2_l = int(shape.part(37).y * 0.9), int(shape.part(40).y * 1.1)
    left = [x1_l, x2_l, y1_l, y2_l]

    # right eye
    x1_r, x2_r = int(shape.part(42).x * 0.9), int(shape.part(45).x * 1.1)
    y1_r, y2_r = int(shape.part(43).y * 0.9), int(shape.part(46).y * 1.1)
    right = [x1_r, x2_r, y1_r, y2_r]

    # left_eye = img[y1_l:y2_l, x1_l:x2_l]
    # right_eye = img[y1_r:y2_r, x1_r:x2_r]

    return left, right


def merge1(track_i):
    merged = track_i
    for i in range(len(track_i)):
        path_img = track_i[i][0]
        left_eye, right_eye = get_face_roi(path_img)
        merged[i].append(left_eye)
        merged[i].append(right_eye)
        print("done with " + path_img)
    return merged


def merge2(track, opt_flow):
    full = track
    for i in range(len(full)):
        full[i].append(opt_flow[i])
    return full


total = []
for a in range(1, 4, 1):
    x_old, y_old = get_x_y(1)
    paths = get_paths_1_by_1(1, a)
    x = split(x_old, len(paths))
    y = split(y_old, len(paths))
    circle = mapping(paths, x, y)
    circle = merge1(circle)
    opt = get_opt_flow_1_by_1(1, a)
    circle = merge2(circle, opt)
    for b in range(len(circle)):
        total.append(circle[b])
for c in range(1, 7, 1):
    x_old, y_old = get_x_y(2)
    paths = get_paths_1_by_1(2, c)
    x = split(x_old, len(paths))
    y = split(y_old, len(paths))
    rect = mapping(paths, x, y)
    rect = merge1(rect)
    opt = get_opt_flow_1_by_1(2, c)
    rect = merge2(rect, opt)
    for d in range(len(rect)):
        total.append(rect[d])

with open("total_dlib.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(total)
