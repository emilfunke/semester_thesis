import cv2

# opencv face and eye detection of pictures
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

img = cv2.imread("camera_output/rect1/000636_fc_1699_processed.jpg", 0)
face = face_cascade.detectMultiScale(img, 1.3, 6)

for (x, y, w, h) in face:
    cv2.rectangle(img, (y, x), (y + h, x + w), (255, 0, 0), 5)
    roi_img = img[x: x + w, y: y + h]
    eyes = eye_cascade.detectMultiScale(roi_img, 1.05, 100)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_img, [ex, ey], [ex + ew, ey + eh], (0, 255, 0), 5)

cv2.imshow('test', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
