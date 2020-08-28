from __future__ import print_function, division

import torch
import pandas as pd
from skimage import io
import numpy as np
from torch.utils.data import Dataset


class HandsDataset(Dataset):
    def __init__(self, csv_file, rows=None, transform=None, img_size=256):
        # super.__init__(self)
        self.frame = pd.read_csv(csv_file, header=None, nrows=rows)
        self.transform = transform
        self.resize_factor = None
        self.in_shape = img_size

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # get data from frame
        img_name = self.frame.iloc[idx, 0]
        image = io.imread(img_name)
        points = self.frame.iloc[idx, 1:]
        points = np.asarray(points)
        points = points.astype('float')
        points = torch.from_numpy(points)
        points = points.reshape(points.numel())

        if self.transform:
            image = self.transform(image)   # Transform image
            if self.resize_factor is None:
                self.resize_factor = image.shape[-1]/self.in_shape
            points *= self.resize_factor

        # normalize points data in [-1, 1]
        points /= self.in_shape * self.resize_factor / 2
        points -= 1
        sample = {'image': image, 'points': points}

        return sample
