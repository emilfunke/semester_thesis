import cv2

# opencv face and eye detection of pictures
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

img = cv2.imread("camera_output/circle3/000047_fc_153_processed.jpg", 0)
face = face_cascade.detectMultiScale(img, 1.05, 4)

for (x, y, w, h) in face:
    cv2.rectangle(img, (y, x), (y + h, x + w), (255, 0, 0), 5)
    roi_img = img[x: x + w, y: y + h]
    eyes = eye_cascade.detectMultiScale(roi_img, 1.05, 150)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_img, [ex, ey], [ex + ew, ey + eh], (0, 255, 0), 5)
    if len(eyes) == 2:
        eyes_x_low = eyes[1][0] if eyes[1][0] < eyes[0][0] else eyes[0][0]
        eyes_y_low = eyes[1][1] if eyes[1][1] < eyes[0][1] else eyes[0][1]
        eyes_x_high = eyes[0][0] if eyes[1][0] < eyes[0][0] else eyes[1][0]
        eyes_y_high = eyes[0][1] if eyes[1][1] < eyes[0][1] else eyes[1][1]
        eyes_w = eyes[1][2] if eyes[1][2] > eyes[0][2] else eyes[0][2]
        eyes_h = eyes[1][3] if eyes[1][3] > eyes[0][3] else eyes[0][3]
    else:
        eyes_x_low, eyes_y_low, eyes_x_high, eyes_y_high, eyes_w, eyes_h = 0, 0, 0, 0, 0, 0
    roi = [eyes_x_low, eyes_y_low, eyes_x_high, eyes_y_high, eyes_w, eyes_h]

    cv2.rectangle(roi_img, [roi[0], roi[1]], [roi[2]+roi[4], roi[3]+roi[5]], (0, 0, 255), 1)

cv2.imshow('roi_img', roi_img)
#cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
