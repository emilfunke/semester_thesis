from __future__ import print_function, division
import os
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
import time
from tqdm import tqdm


class GazeEstimationDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.eyes_frame = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.eyes_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.eyes_frame.iloc[idx, 0]
        image = io.imread(img_name)
        image = image/255
        rgb_img = np.repeat(image[..., np.newaxis], 3, -1)
        #print(rgb_img.shape, "image shape")
        face_img_coor = np.fromstring(self.eyes_frame.iloc[idx, 3][1:int(len(self.eyes_frame.iloc[idx, 3]) - 1)],
                                      sep=',',
                                      dtype=int)
        roi_eyes_coor = np.fromstring(self.eyes_frame.iloc[idx, 4][1:int(len(self.eyes_frame.iloc[idx, 4]) - 1)],
                                      sep=',', dtype=int)
        face_img = rgb_img[face_img_coor[0]: face_img_coor[0] + face_img_coor[2],
                   face_img_coor[1]: face_img_coor[1] + face_img_coor[3], :]

        opt_flow_name = self.eyes_frame.iloc[idx, 5]
        opt_flow = pd.read_csv(opt_flow_name)
        sample = {'name': img_name, 'face': face_img, 'face_coor': face_img_coor, 'eyes_coor': roi_eyes_coor,
                  'x': self.eyes_frame.iloc[idx, 1], 'y': self.eyes_frame.iloc[idx, 2]}
        '''opt_flow': opt_flow'''
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
        # if isinstance(self.output_size, int):
        #    if h > w:
        #        new_h, new_w = self.output_size * h / w, self.output_size
        #    else:
        #        new_h, new_w = self.output_size, self.output_size * w / h
        # else:
        new_h, new_w = self.output_size, self.output_size
        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        for i in range(2):
            eyes[i] = eyes[i] * new_w / w
        for i in range(2, 4, 1):
            eyes[i] = eyes[i] * new_h / h
        eyes[4] = eyes[4] * new_w / w
        eyes[5] = eyes[5] * new_h / h

        x, y = sample['x'], sample['y']
        # opt_flow = sample['opt_flow']

        return {'face': img, 'eyes_coor': eyes, 'x': x, 'y': y}

    '''opt_flow': opt_flow'''


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
            eyes[i] = eyes[i] - left
        for i in range(2, 4, 1):
            eyes[i] = eyes[i] - top
        eyes[4] = eyes[4] + top
        eyes[5] = eyes[5] + left

        x, y = sample['x'], sample['y']
        # opt_flow = sample['opt_flow']

        return {'face': image, 'eyes_coor': eyes, 'x': x, 'y': y}

    '''opt_flow': opt_flow'''


class ToTensor(object):
    def __call__(self, sample):
        image, eyes = sample['face'], sample['eyes_coor']
        x, y = (sample['x'] + 800) / 1600, (sample['y'] + 800) / 1600
        # opt_flow = sample['opt_flow']
        image = image.transpose((2, 0, 1))
        return {'face': torch.from_numpy(image).type(torch.DoubleTensor),
                'eyes_coor': torch.from_numpy(eyes).type(torch.DoubleTensor),
                'gt_coor': torch.tensor([x, y]).type(torch.DoubleTensor)}

    '''opt_flow': torch.from_numpy(opt_flow.values)'''


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


def show_eyes_batch(sample_batched):
    images_batch, eyes_batch = sample_batched['face'], sample_batched['eyes_coor']
    print(eyes_batch)
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        for j in range(len(eyes_batch[i])):
            x, y = eyes_batch[i][0], eyes_batch[i][2]
            w = eyes_batch[i][4] + eyes_batch[i][1] - eyes_batch[i][0]
            h = eyes_batch[i][5] + eyes_batch[i][3] - eyes_batch[i][2]
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax = plt.subplot()
            ax.add_patch(rect)


dataset = GazeEstimationDataset(csv_file="full_face/total.csv", root_dir="")
'''
face_frame = pd.read_csv('total.csv')

n = 0
img_path = face_frame.iloc[n, 0]
face_coor = np.fromstring(face_frame.iloc[n, 3][1:len(face_frame.iloc[n, 3]) - 1], sep=',', dtype=int)
roi_eyes = np.fromstring(face_frame.iloc[n, 4][1:len(face_frame.iloc[n, 4]) - 1], sep=',', dtype=int)



for i in range(870, 871, 1):
    sample = dataset[i]
    print(i, sample['x'])
    show_roi(sample['name'], sample['face_coor'], sample['eyes_coor'])
'''

transformed_dataset = GazeEstimationDataset(csv_file="full_face/total.csv", root_dir="",
                                            transform=transforms.Compose([Rescale(256), ToTensor()]))

'''
scale = Rescale(256)
# crop = RandomCrop(1024)
# composed = transforms.Compose([Rescale(256), RandomCrop(255)])

sample = dataset[65]
fig = plt.figure()
for i, tsfrm in enumerate([scale]):
    temp = sample
    transformed_sample = tsfrm(temp)
    show_eyes(transformed_sample['face'], transformed_sample['eyes_coor'])
    plt.pause(0.01)
    plt.show()
'''
# print(transformed_dataset.__getitem__(1))
batch_size = 20
dataloader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

'''
for i_batch, sample_batched in enumerate(dataloader):
    # print(i_batch, sample_batched['face'].size(), sample_batched['eyes_coor'])

    if i_batch == 3:
        plt.figure()
        show_eyes_batch(sample_batched)
        plt.show()
        break
'''

net = Net()
net = net.double()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

n = 5
for epoch in range(n):
    with tqdm(dataloader, unit="batch") as tepoch:
        running_loss = 0.0
        for data in tepoch:
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

            correct = (abs(output - labels)).sum().item()
            accuracy = correct / batch_size

            tepoch.set_postfix(loss=loss.item(), accuracy=100*accuracy)
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
