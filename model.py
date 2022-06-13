import pandas as pd
import os
import cv2
from datetime import datetime, timezone
from dateutil import tz


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
        roi_img = img[x: x + w, y: y + h]
        eyes = eye_cascade.detectMultiScale(roi_img, 1.05, 100)
        if eyes.size != 8:
            print("fuck")
        else:
            eyes_x_low = eyes[1][0] if eyes[1][0] < eyes[0][0] else eyes[0][0]
            eyes_y_low = eyes[1][1] if eyes[1][1] < eyes[0][1] else eyes[0][1]
            eyes_x_high = eyes[0][0] if eyes[1][0] < eyes[0][0] else eyes[1][0]
            eyes_y_high = eyes[0][1] if eyes[1][1] < eyes[0][1] else eyes[1][1]
            eyes_w = eyes[1][2] if eyes[1][2] > eyes[0][2] else eyes[0][2]
            eyes_h = eyes[1][3] if eyes[1][3] > eyes[0][3] else eyes[0][3]
        roi = [eyes_x_low, eyes_y_low, eyes_x_high, eyes_y_high, eyes_w, eyes_h]
    return roi_img, roi


def show_eyes(roi_img, roi):
    cv2.rectangle(roi_img, [roi[0], roi[1]], [roi[2] + roi[4], roi[3] + roi[5]], (0, 0, 255), 1)
    cv2.imshow('roi_img', roi_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


# circle 1-3
circle_jpg_name, circle_of, circle_time = {}, {}, {}
for i in range(1, 4, 1):
    path_img_circle = "camera_output/circle" + str(i)
    circle_jpg_name["circle" + str(i)] = [f for f in os.listdir(path_img_circle) if f.endswith('.jpg')]
    circle_of["circle" + str(i)] = [f for f in os.listdir(path_img_circle) if f.endswith('.csv')]
    circle_time["circle" + str(i)] = [get_time(path_img_circle + "/" + g) for g in circle_jpg_name["circle" + str(i)]]

# rect 1-6
rect_jpg_name, rect_of, rect_time = {}, {}, {}
for i in range(1, 7, 1):
    path_img_rect = "camera_output/rect" + str(i)
    rect_jpg_name["rect" + str(i)] = [f for f in os.listdir(path_img_rect) if f.endswith('.jpg')]
    rect_of["rect" + str(i)] = [f for f in os.listdir(path_img_rect) if f.endswith('.csv')]
    rect_time["rect" + str(i)] = [get_time(path_img_rect + "/" + g) for g in rect_jpg_name["rect" + str(i)]]

start = datetime.now()
# get roi_img and roi for all images
roi_img_all, roi_all = {}, {}
for i in range(1, 4, 1):
    path_img_circle = "camera_output/circle" + str(i)
    for name in circle_jpg_name["circle" + str(i)]:
        path = path_img_circle + "/" + name
        roi_img_all[name + " circle" + str(i)] = get_roi_eyes(path_img_circle + "/" + name)[0], path
        roi_all[name + " circle" + str(i)] = get_roi_eyes(path_img_circle + "/" + name)[1], path
    print(start - datetime.now())
mid = datetime.now()
print(mid - start + " total time for circle")
for i in range(1, 7, 1):
    path_img_rect = "camera_output/rect" + str(i)
    for name in circle_jpg_name["rect" + str(i)]:
        path = path_img_rect + "/" + name
        roi_img_all[name + " rect" + str(i)] = get_roi_eyes(path_img_rect + "/" + name)[0], path
        roi_all[name + " rect" + str(i)] = get_roi_eyes(path_img_rect + "/" + name)[1], path
    print(mid - datetime.now())
end = datetime.now()
print(end - mid + " total time rect")
print(end - start + " total time all")

df_circle_time = pd.DataFrame(circle_time["circle3"])
df_circle_time.to_csv('circle_time')
circle1_csv = pd.read_csv("csv/circle1.csv", delimiter=',')

# roi_img, roi = get_roi_eyes(path_img_circle + "/" + circle_jpg_name["circle3"][143])
# show_eyes(roi_img, roi)

print(roi_all)
