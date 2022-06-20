import csv
import numpy as np
import pandas as pd
import os
import cv2
import math


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
    l =  list((arr[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)))
    for i in range(n):
        tot = 0
        for j in range(len(l[i])):
            tot += l[i][j]
        ret.append(tot/len(l[i]))
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
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    face = face_cascade.detectMultiScale(img, 1.05, 4)
    for (x, y, w, h) in face:
        x *= 0.75
        y *= 0.75
        w *= 1.25
        h *= 1.25
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        face_img = img[x: x + w, y: y + h]
        face_img_coor = [x, y, w, h]
        eyes = eye_cascade.detectMultiScale(face_img, 1.05, 100)
        x_low, x_high, y_low, y_high, e_w, e_h = 10000, 0, 10000, 0, 0, 0
        for (ex, ey, ew, eh) in eyes:
            if ex < x_low:
                x_low = ex
            if ex > x_high:
                x_high = ex
            if ey < y_low:
                y_low = ey
            if ey > y_high:
                y_high = ey
            if ew > e_w:
                e_w = ew
            if eh > e_h:
                e_h = ew
        x_low = math.floor(x_low*0.95)
        x_high = math.floor(x_high*1.05)
        y_low = math.floor(y_low*0.95)
        y_high = math.floor(y_high*1.05)
        e_w = math.floor(e_w*1.05)
        e_h = math.floor(e_h*1.05)
        roi_eyes = [x_low, x_high, y_low, y_high, e_w, e_h]
        cv2.rectangle(face_img, [roi_eyes[0], roi_eyes[2]], [roi_eyes[1] + roi_eyes[4], roi_eyes[3] + roi_eyes[5]],
                      (0, 0, 255), 1)
    return face_img, face_img_coor, roi_eyes


def merge1(track_i):
    merged = track_i
    for i in range(len(track_i)):
        path_img = track_i[i][0]
        face_img, face_img_coor, roi_eyes = get_face_roi(path_img)
        if roi_eyes[0] == 9500:
            face_img_coor = merged[i-1][3]
            roi_eyes = merged[i-1][4]
        # merged[i].append(face_img)
        merged[i].append(face_img_coor)
        merged[i].append(roi_eyes)
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

with open("total.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(total)
