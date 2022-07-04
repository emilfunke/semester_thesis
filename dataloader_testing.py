from __future__ import print_function, division
import os
import time

import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import Dataset, DataLoader, default_convert
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
from tqdm import tqdm


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
        # print(img_norm.shape)

        eyes_roi = np.fromstring(self.frame.iloc[idx, 4][1:int(len(self.frame.iloc[idx, 4]) - 1)],
                                      sep=',', dtype=int)
        x_eye, y_eye = eyes_roi[0], eyes_roi[2]
        w = eyes_roi[1] + eyes_roi[4]
        h = eyes_roi[3] + eyes_roi[5]
        eyes_roi = [x_eye, y_eye, w, h]

        eyes_img = img_norm[y_eye: y_eye + h, x_eye: x_eye + w]

        x_gt = (self.frame.iloc[idx, 1] + 800) / 1600
        y_gt = (self.frame.iloc[idx, 2] + 800) / 1600

        sample = {'name': img_name, 'image': img_norm, 'eyes_roi': eyes_roi, 'eyes_img': eyes_img,
                  'x': x_gt, 'y': y_gt}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, eyes = sample['image'], sample['eyes_roi']

        h, w = image.shape[:2]
        # if isinstance(self.output_size, int):
        #    if h > w:
        #        new_h, new_w = self.output_size * h / w, self.output_size
        #    else:
        #        new_h, new_w = self.output_size, self.output_size * w / h
        # else:
        new_h, new_w = self.output_size, self.output_size
        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        eyes[0] = eyes[0] * new_w / w
        eyes[2] = eyes[1] * new_h / h
        eyes[2] = eyes[2] * new_w / w
        eyes[3] = eyes[3] * new_h / h

        x, y = sample['x'], sample['y']
        # opt_flow = sample['opt_flow']

        return {'image': img, 'eyes_roi': eyes, 'x': x, 'y': y}


class ToTensor(object):
    def __call__(self, sample):
        image, eyes, x, y = sample['image'], sample['eyes_roi'], sample['x'], sample['y']
        # opt_flow = sample['opt_flow']
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).type(torch.DoubleTensor),
                'eyes_roi': torch.tensor(eyes).type(torch.DoubleTensor),
                'gt_coor': torch.tensor([x, y]).type(torch.DoubleTensor)}


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(59536, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


dataset = GazeEstimationDataset(csv_file="full_face/total.csv", root_dir="")

transformed_dataset = GazeEstimationDataset(csv_file="full_face/total.csv", root_dir="",
                                            transform=transforms.Compose([Rescale(256), ToTensor()]))

dataloader = DataLoader(transformed_dataset, batch_size=20, shuffle=True, num_workers=0)


net = Net()
net = net.double()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

n = 5
for epoch in range(n):
    with tqdm(dataloader, unit="batch") as tepoch:
        running_loss = 0.0
        for i, data in tepoch:
            # print(data['face'].shape)
            tepoch.set_description(f"Epoch {epoch}")
            inputs = data['face']
            # data['eyes_coor']
            # data['opt_flow']
            labels = data['gt_coor']

            optimizer.zero_grad()

            output = net(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            tepoch.set_postfix(loss=loss.item(), accuracy=(abs(output-labels))/(output+labels))
            time.sleep(0.1)

            """
            running_loss += loss.item()
            if i % 20 == 19:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20:.3f}')
                # print(f'[{"labels"}, {labels}]', f'[{"output"}, {output}]')
                print(f'[{"accuracy"}, {abs(output / labels)}]')
                # , f'[{"output"}, {output}]'
                running_loss = 0.0
            """


print('done')
path = '../trained.pth'
torch.save(net.state_dict(), path)
