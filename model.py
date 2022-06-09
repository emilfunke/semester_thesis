import pandas as pd
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
circle_jpg, circle_of, circle_time = {}, {}, {}
for i in range(1, 4, 1):
    path_img_circle = "camera_output/circle" + str(i)
    circle_jpg["circle" + str(i)] = [f for f in os.listdir(path_img_circle) if f.endswith('.jpg')]
    circle_of["circle" + str(i)] = [f for f in os.listdir(path_img_circle) if f.endswith('.csv')]
    circle_time["circle" + str(i)] = [get_time(path_img_circle + "/" + g) for g in circle_jpg["circle" + str(i)]]

# rect 1-6
rect_jpg, rect_of, rect_time = {}, {}, {}
for i in range(1, 7, 1):
    path_img_rect = "camera_output/rect" + str(i)
    rect_jpg["rect" + str(i)] = [f for f in os.listdir(path_img_rect) if f.endswith('.jpg')]
    rect_of["rect" + str(i)] = [f for f in os.listdir(path_img_rect) if f.endswith('.csv')]
    rect_time["rect" + str(i)] = [get_time(path_img_rect + "/" + g) for g in rect_jpg["rect" + str(i)]]

df_circle_time = pd.DataFrame(circle_time["circle3"])
df_circle_time.to_csv('circle_time')
circle1_csv = pd.read_csv("csv/circle1.csv", delimiter=',')



