import pandas as pd
import os
import cv2
from datetime import datetime, timezone
from dateutil import tz
import numpy as np


# get date and time of picture creation
def get_time(path_img):
    mtime = os.path.getmtime(path_img)
    time_utc = datetime.fromtimestamp(mtime, tz=timezone.utc)
    to_zone = tz.tzlocal()
    time = time_utc.astimezone(to_zone)
    return time


def get_roi_eyes(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    img = cv2.imread(image, 0)
    face = face_cascade.detectMultiScale(img, 1.05, 4)

    for (x, y, w, h) in face:
        roi_img_ret = img[x: x + w, y: y + h]
        eyes = eye_cascade.detectMultiScale(roi_img_ret, 1.05, 100)
        if len(eyes) == 0 or eyes.size != 8:
            eyes_x_low, eyes_y_low, eyes_x_high, eyes_y_high, eyes_w, eyes_h = 0, 0, 0, 0, 0, 0
            show_bad_img(roi_img_ret)
        else:
            eyes_x_low = eyes[1][0] if eyes[1][0] < eyes[0][0] else eyes[0][0]
            eyes_y_low = eyes[1][1] if eyes[1][1] < eyes[0][1] else eyes[0][1]
            eyes_x_high = eyes[0][0] if eyes[1][0] < eyes[0][0] else eyes[1][0]
            eyes_y_high = eyes[0][1] if eyes[1][1] < eyes[0][1] else eyes[1][1]
            eyes_w = eyes[1][2] if eyes[1][2] > eyes[0][2] else eyes[0][2]
            eyes_h = eyes[1][3] if eyes[1][3] > eyes[0][3] else eyes[0][3]
        roi_ret = [eyes_x_low, eyes_y_low, eyes_x_high, eyes_y_high, eyes_w, eyes_h]
    return roi_img_ret, roi_ret


def show_eyes(roi_img, roi):
    cv2.rectangle(roi_img, [roi[0], roi[1]], [roi[2] + roi[4], roi[3] + roi[5]], (0, 0, 255), 1)
    cv2.imshow('roi_img', roi_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


def show_bad_img(image):
    cv2.imshow('bad image no 2 eyes found', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


def get_all_img_paths(r):
    paths = {}
    for subdir, dirs, files in os.walk(r):
        for file in files:
            if file.endswith(".jpg"):
                paths.setdefault(subdir, [])
                paths[subdir].append(os.path.join(subdir, file))
    return paths


def get_all_x_known(r):
    x = []
    for name in os.listdir(r):
        if name.endswith("e.csv") or name.endswith("t.csv"):
            key = name[:len(name) - 4]
            df_x = pd.read_csv(r + "/" + name)
            x.setdefault(key, [])
            x[key].append(df_x.iloc[:, 1])
    return x


def get_dict():
    dict_preprocessed = {}
    for i in range(1, 3, 1):
        start = datetime.now()
        path_noname = "camera_output/circle" + str(i) + "/"
        for name in os.listdir(path_noname):
            if name.endswith(".jpg"):
                path = path_noname + name
                roi_img, roi = get_roi_eyes(path)
                dict_preprocessed.setdefault(path, [])
                dict_preprocessed[path].append(roi_img)
                dict_preprocessed[path].append(roi)
                dict_preprocessed[path].append(path[:len(path) - 3] + "csv")
                print(path, " done")
        end = datetime.now()
        print("time for circle" + str(i) + " ", end - start)
        return dict_preprocessed


def create_csv():
    df_circle1 = pd.read_csv("csv/circle1.csv")
    df_circle_xy = df_circle1.iloc[:, :2]
    df_circle_xy.to_csv("csv/circle.csv")
    df_circle1 = pd.read_csv("csv/rect1.csv")
    df_circle_xy = df_circle1.iloc[:, :2]
    df_circle_xy.to_csv("csv/rect.csv")
    return


'''
# circle 1-3
circle_jpg_name, circle_of, circle_time = {}, {}, {}
for i in range(1, 4, 1):
    path_img_circle = "camera_output/circle" + str(i)
    circle_jpg_name[path_img_circle] = [f for f in os.listdir(path_img_circle) if f.endswith('.jpg')]
    circle_of[path_img_circle] = [f for f in os.listdir(path_img_circle) if f.endswith('.csv')]
    circle_time[path_img_circle] = [get_time(path_img_circle + "/" + g) for g in circle_jpg_name["circle" + str(i)]]

# rect 1-6
rect_jpg_name, rect_of, rect_time = {}, {}, {}
for i in range(1, 7, 1):
    path_img_rect = "camera_output/rect" + str(i)
    rect_jpg_name[path_img_rect] = [f for f in os.listdir(path_img_rect) if f.endswith('.jpg')]
    rect_of[path_img_rect] = [f for f in os.listdir(path_img_rect) if f.endswith('.csv')]
    rect_time[path_img_rect] = [get_time(path_img_rect + "/" + g) for g in rect_jpg_name["rect" + str(i)]]
'''
'''
circle_time_csv, rect_time_csv = {}, {}
for i in range(1, 4, 1):
    circle_time_csv["circle" + str(i)] = pd.read_csv("csv/circle" + str(i) + ".csv", delimiter=',')

for i in range(1, 4, 1):
    rect_time_csv["rect" + str(i)] = pd.read_csv("csv/rect" + str(i) + ".csv", delimiter=',')

print(circle_time_csv)

df_circle_time_img = pd.DataFrame.from_dict(circle_time, orient='index')
df_circle_time_img.to_csv('circle_time_img')
df_rect_time_img = pd.DataFrame.from_dict(rect_time, orient='index')
df_rect_time_img.to_csv('rect_time_img')

df_circle_time_csv = pd.DataFrame.from_dict(circle_time_csv["circle1"], orient='index')
df_circle_time_csv.to_csv('circle_time_csv')
df_rect_time_csv = pd.DataFrame.from_dict(rect_time_csv["rect1"], orient='index')
df_rect_time_csv.to_csv('rect_time_csv')
'''

'''
# extracting the rio from each picture and storing in one large dictionary:
# get roi_img and roi for all images
roi_img_all, roi_all = {}, {}
for i in range(1, 4, 1):
    path_img_circle = "camera_output/circle" + str(i)
    for name in circle_jpg_name["circle" + str(i)]:
        path = path_img_circle + "/" + name
        roi_img_all[name + " circle" + str(i)] = get_roi_eyes(path_img_circle + "/" + name)[0], path
        roi_all[name + " circle" + str(i)] = get_roi_eyes(path_img_circle + "/" + name)[1], path
for i in range(1, 7, 1):
    path_img_rect = "camera_output/rect" + str(i)
    for name in circle_jpg_name["rect" + str(i)]:
        path = path_img_rect + "/" + name
        roi_img_all[name + " rect" + str(i)] = get_roi_eyes(path_img_rect + "/" + name)[0], path
        roi_all[name + " rect" + str(i)] = get_roi_eyes(path_img_rect + "/" + name)[1], path
df_roi_img_all = pd.DataFrame.from_dict(roi_img_all, orient='index')
df_roi_all = pd.DataFrame.from_dict(roi_all, orient='index')

# roi_img, roi = get_roi_eyes(path_img_circle + "/" + circle_jpg_name["circle3"][143])
# show_eyes(roi_img, roi)
'''

all_img_paths = get_all_img_paths(r="camera_output")
all_x_known = get_all_x_known(r="csv")
print(all_x_known)

l = len(all_img_paths["camera_output\\circle1"])
c_data = list(all_x_known.items())
c = np.array(c_data)
print(c)
circle = np.array_split(c, l)

