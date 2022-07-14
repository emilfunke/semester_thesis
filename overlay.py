from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.colors
from matplotlib.transforms import Bbox
import numpy as np
import csv

# DataPath
DATA_PATH = "C:/st_camera/s6g3-sdk/OutputData/"
# Flow Subfolder
FLOW_SUBPATH = "Flow/"
# Frames Subfolder
FRAMES_SUBPATH = "Frames/"

OVERLAY_SUBPATH = "overlay_colored/"

# SUBSAMPLING = 2
NUMBER_FRAMES = 9806

LARGEST_DISPLACEMENT = 50

def vector_to_rgb(angle, absolute=40):
    max_abs = LARGEST_DISPLACEMENT

    # normalize angle
    angle = (angle-5*np.pi/8) % (2 * np.pi)

    return matplotlib.colors.hsv_to_rgb((angle / 2 / np.pi, 
                                         absolute / max_abs, 
                                         absolute / max_abs))


for i in range(3636, NUMBER_FRAMES, 1):
    print("Processing Frame", i)
    frame = Image.open(DATA_PATH+FRAMES_SUBPATH+"frame_"+str(i)+'.png').convert('L')
    
    with open(DATA_PATH+FLOW_SUBPATH+"flow_"+str(i)+'.csv') as f:
        x_start = []
        y_start = []
        dx = []
        dy = []
        reader = csv.reader(f)
        for idx,row in enumerate(reader):
            if idx > 0:
                x_start.append(int(row[0]))
                y_start.append(int(row[1]))
                dx.append(int(row[2]))
                dy.append(int(row[3]))
    x_start = np.array(x_start)
    y_start = np.array(y_start)
    dx = np.array(dx)
    dy = np.array(dy)

    angles = np.arctan2(dx, dy)
    lengths = np.sqrt(np.square(dx) + np.square(dy))

    c = np.array(list(map(vector_to_rgb, angles.flatten())))

    plt.clf()
    plt.quiver(x_start, y_start, dx, dy, angles="xy", scale=1.0, scale_units="xy",color=c, width=0.005)
    plt.imshow(frame,interpolation='none',cmap='gray')
    plt.axis('off')
    # Setting super weird dpi to match original resoltuion
    plt.savefig(DATA_PATH+OVERLAY_SUBPATH+"frame_"+str(i)+'.jpg',bbox_inches='tight',pad_inches=0,dpi=369.3)
