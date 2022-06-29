import os
import subprocess
from datetime import datetime

print(datetime.now())
path_org = os.getcwd()
file_ex = "S6G3_SDK_opencv.exe"
path_ex = "C:/st_camera/S6G3_SDK-v1.0.0/Build/Release/"

p = subprocess.Popen([path_ex + file_ex], cwd=path_ex)
p.wait()

print(datetime.now())
