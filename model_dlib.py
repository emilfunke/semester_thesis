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
        self.frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self, idx):
        return int(len(self.frame))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.frame.iloc[idx, 0]

        image = cv2.imread(img_name)
        img_norm = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        #print(img_norm.shape)

        eyes_roi = self.frame.iloc[idx, 4]
        x, y = eyes_roi[0], eyes_roi[2]
        w = eyes_roi[1] + eyes_roi[4]
        h = eyes_roi[3] + eyes_roi[5]
        eyes_roi = [x, y, w, h]

        eyes_img = img_norm[x: x+w, y: y+h]

        sample = {'image': img_norm, 'eyes_roi' : eyes_roi, 'eyes_img': eyes_img}

        if self.transform:
            sample = self.transform(sample)

        return sample


dataset = GazeEstimationDataset(csv_file="full_face/total.csv", root_dir="")

sample = dataset[0]

norm, eyes = sample['image'], sample['eyes_img']

