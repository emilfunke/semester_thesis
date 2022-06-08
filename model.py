import numpy as np
import cv2
from PIL import Image
from PIL.ExifTags import TAGS

# opencv face and eye detection of pictures

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

image_name = "000000_fc_46_processed.jpg"
image = Image.open("camera_output/2022-05-27_12-27-07_spiral1/" + image_name)
print(image.filename)
exif_data = image.getexif()
print(exif_data)
for tag_id in exif_data:
    tag = TAGS.get(tag_id, tag_id)
    data = exif_data.get(tag_id)
    if isinstance(data, bytes):
        data = data.decode()
    print("pisser")
