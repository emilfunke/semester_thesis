import numpy as np
import os
import cv2
from datetime import datetime, timezone
from dateutil import tz

# opencv face and eye detection of pictures
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# get date and time of picture creation from metadata
image_name = "000000_fc_46_processed.jpg"
mtime = os.path.getmtime("camera_output/2022-05-27_12-27-07_spiral1/" + image_name)
time_utc = datetime.fromtimestamp(mtime, tz=timezone.utc)
to_zone = tz.tzlocal()
time = time_utc.astimezone(to_zone)
