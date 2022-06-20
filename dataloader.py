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
        face_img_coor = np.fromstring(self.eyes_frame.iloc[idx, 3][1:int(len(self.eyes_frame.iloc[idx, 3]) - 1)],
                                      sep=',',
                                      dtype=int)
        roi_eyes_coor = np.fromstring(self.eyes_frame.iloc[idx, 4][1:int(len(self.eyes_frame.iloc[idx, 4]) - 1)],
                                      sep=',', dtype=int)
        face_img = image[face_img_coor[0]: face_img_coor[0] + face_img_coor[2], face_img_coor[1]: face_img_coor[1]
                                                                                                  + face_img_coor[3]]
        sample = {'name': img_name, 'face': face_img, 'face_coor': face_img_coor, 'eyes_coor': roi_eyes_coor,
                  'x': self.eyes_frame.iloc[idx, 1], 'y': self.eyes_frame.iloc[idx, 2]}
        if self.transform:
            sample = self.transform(sample)
        return sample


class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, eyes = sample['face'], sample['eyes_coor']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        for i in range(2):
            eyes[i] = eyes[i] * new_w / w
        for i in range(2, 4, 1):
            eyes[i] = eyes[i] * new_h / h
        eyes[4] = eyes[4] * new_w / w
        eyes[5] = eyes[5] * new_h / h

        return {'face': img, 'eyes_coor': eyes}


class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, eyes = sample['face'], sample['eyes_coor']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]

        for i in range(2):
            eyes[i] = eyes[i] + left
        for i in range(2, 4, 1):
            eyes[i] = eyes[i] + top
        eyes[4] = eyes[4] + top
        eyes[5] = eyes[5] * left

        return {'face': image, 'eyes_coor': eyes}


class ToTensor(object):
    def __call__(self, sample):
        image, eyes = sample['face'], sample['eyes_coor']
        image = image.transpose((2, 0, 1))
        return {'face': torch.from_numpy(image), 'eyes_coor': torch.from_numpy(eyes)}


def show_roi(path, face_c, roi):
    image = cv2.imread(path)
    face = image[face_c[0]: face_c[0] + face_c[2], face_c[1]: face_c[1] + face_c[3]]
    cv2.rectangle(face, [roi[0], roi[2]], [roi[1] + roi[4], roi[3] + roi[5]], (0, 0, 255), 1)
    cv2.imshow('annotated face', face)
    # cv2.imshow("original", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_eyes(image, eyes):
    plt.imshow(image)
    x = eyes[0]
    y = eyes[2]
    w = eyes[4] + eyes[1] - eyes[0]
    h = eyes[5] + eyes[3] - eyes[2]
    rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
    ax = plt.subplot()
    ax.add_patch(rect)


dataset = GazeEstimationDataset(csv_file="total.csv", root_dir="")
'''
face_frame = pd.read_csv('total.csv')

n = 0
img_path = face_frame.iloc[n, 0]
face_coor = np.fromstring(face_frame.iloc[n, 3][1:len(face_frame.iloc[n, 3]) - 1], sep=',', dtype=int)
roi_eyes = np.fromstring(face_frame.iloc[n, 4][1:len(face_frame.iloc[n, 4]) - 1], sep=',', dtype=int)


# circle 3 images in the beginning not right
for i in range(870, 871, 1):
    sample = dataset[i]
    print(i, sample['x'])
    show_roi(sample['name'], sample['face_coor'], sample['eyes_coor'])
'''

scale = Rescale(256)
crop = RandomCrop(1000)
composed = transforms.Compose([Rescale(256), RandomCrop(255)])

sample = dataset[65]
fig = plt.figure()
for i, tsfrm in enumerate([scale, crop, composed]):
    transformed_sample = tsfrm(sample)
    print(transformed_sample['eyes_coor'])
    show_eyes(transformed_sample['face'], transformed_sample['eyes_coor'])
    plt.pause(0.01)
    plt.show()
