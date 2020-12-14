from torchvision import datasets
import os
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
import cv2
from PIL import Image

class ChannelDataset(datasets.ImageFolder):

    def __init__(self, root, transform, domain_num=3, train=True, gait=False):
        super(ChannelDataset, self).__init__(root, transform)
        self.domain_num = domain_num
        self.labels = np.array(self.imgs)[:, 1]
        self.data = np.array(self.imgs)[:, 0]
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        self.class_num = len(self.classes)
        class_name = []
        for s in self.samples:
            filename = os.path.basename(s[0])
            class_name.append(filename.split('_')[0])
        self.class_name = np.asarray(class_name)

        cams = []
        for s in self.samples:
            cams.append(self._get_cam_id(s[0]))
        self.cams = np.asarray(cams)
        self.transform = transform
        self.train = train
        self.root = root
        self.gait = gait
    def _get_cam_id(self, path):
        filename = os.path.basename(path)
        if 'foot' in self.root or 'gait' in self.root:
            camera_id = 1
        elif 'msmt' in self.root:
            camera_id = filename[9:11]
        else:
            camera_id = filename.split('c')[1][0]
        return int(camera_id) - 1

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index].item()
        img = default_loader(img)
        # The index_channel is used to shuffle channels of the original image
        index_channel = [[0, 1, 2],
                         [0, 2, 1],
                         [1, 0, 2],
                         [1, 2, 0],
                         [2, 0, 1],
                         [2, 1, 0]]
        order = np.random.randint(self.domain_num)
        img = img.split()
        img = Image.merge('RGB', (img[index_channel[order][0]], img[index_channel[order][1]],
                                  img[index_channel[order][2]]))

        if self.transform is not None:
            img = self.transform(img)

        # #for gait
        if self.gait:
            return img, int(label), order
        # #for re-ID
        else:
            return img, int(label) + self.class_num * order, order

    def __len__(self):
        return len(self.imgs)





