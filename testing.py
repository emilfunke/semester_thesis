import math
import cv2

img_path = "camera_output/circle1/000049_fc_155_processed.jpg"

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
    cv2.rectangle(face_img, [roi_eyes[0], roi_eyes[2]], [roi_eyes[1] + roi_eyes[4], roi_eyes[3] + roi_eyes[5]], (0, 0, 255), 1)

cv2.imshow('face', face_img)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()