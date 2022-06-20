from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2


class GazeEstimationDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.eyes_frame = pd.read_csv('total.csv')

    def __len__(self):
        return len(self.eyes_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.eyes_frame.iloc[idx, 0]
        image = io.imread(img_name)
        face_img_coor = np.fromstring(self.eyes_frame.iloc[idx, 3][1:int(len(self.eyes_frame.iloc[idx, 3]) - 1)], sep=',',
                                      dtype=int)
        roi_eyes_coor = np.fromstring(self.eyes_frame.iloc[idx, 4][1:int(len(self.eyes_frame.iloc[idx, 4]) - 1)], sep=',', dtype=int)
        face_img = image[face_img_coor[0]: face_img_coor[0] + face_img_coor[2], face_img_coor[1]: face_img_coor[1]
                         + face_img_coor[3]]
        sample = {'name': img_name, 'face': face_img, 'face_coor': face_img_coor, 'eyes_coor': roi_eyes_coor}
        if self.transform:
            sample = self.transform(sample)
        return sample


def show_roi(path, face_c, roi):
    image = cv2.imread(path)
    face = image[face_c[0]: face_c[0] + face_c[2], face_c[1]: face_c[1] + face_c[3]]
    cv2.rectangle(face, [roi[0], roi[2]], [roi[1] + roi[4], roi[3] + roi[5]], (0, 0, 255), 1)
    cv2.imshow('annotated face', face)
    # cv2.imshow("original", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


face_frame = pd.read_csv('total.csv')

n = 0
img_path = face_frame.iloc[n, 0]
face_coor = np.fromstring(face_frame.iloc[n, 3][1:len(face_frame.iloc[n, 3]) - 1], sep=',', dtype=int)
roi_eyes = np.fromstring(face_frame.iloc[n, 4][1:len(face_frame.iloc[n, 4]) - 1], sep=',', dtype=int)

dataset = GazeEstimationDataset(csv_file="total.csv", root_dir="")

for i in range(100):
    sample = dataset[i]
    print(i, sample['face'].shape)
    show_roi(sample['name'], sample['face_coor'], sample['eyes'])
