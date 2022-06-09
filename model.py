import numpy as np
import os
import cv2
from datetime import datetime, timezone
from dateutil import tz


# get date and time of picture creation from metadata
def get_time(path_img):
    mtime = os.path.getmtime(path_img)
    time_utc = datetime.fromtimestamp(mtime, tz=timezone.utc)
    to_zone = tz.tzlocal()
    time = time_utc.astimezone(to_zone)
    return time


# circle 1-3
circle = {}
circle_time = {}
for i in range(1, 4, 1):
    path_img_circle = "camera_output/circle" + str(i)
    circle["circle" + str(i)] = [f for f in os.listdir(path_img_circle) if f.endswith('.jpg')]
    circle_time["circle" + str(i)] = [get_time(path_img_circle + "/" + g) for g in circle["circle" + str(i)]]

# rect 1-6
rect = {}
rect_time = {}
for i in range(1, 7, 1):
    path_img_rect = "camera_output/rect" + str(i)
    rect["rect" + str(i)] = [f for f in os.listdir(path_img_rect) if f.endswith('.jpg')]
    rect_time["rect" + str(i)] = [get_time(path_img_rect + "/" + g) for g in rect["rect" + str(i)]]


# opencv face and eye detection of pictures
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
