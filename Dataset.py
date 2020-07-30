from __future__ import print_function, division

import torch
import pandas as pd
from skimage import io
import numpy as np
from torch.utils.data import Dataset


class HandsDataset(Dataset):
    def __init__(self, csv_file, rows=None, transform=None):
        self.frame = pd.read_csv(csv_file, header=None, nrows=rows, skiprows=150000)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.frame.iloc[idx, 0]
        image = io.imread(img_name)
        points = self.frame.iloc[idx, 1:]
        points = np.asarray(points)
        points = points.astype('float').reshape(-1, 21, 3)

        if self.transform:
            # TODO: Optimizeaza operatiile de transformare
            # image = Image.fromarray(image)
            image = self.transform(image)
            # image = np.asarray(image)/(255 / 2) - 1
        sample = {'image': image, 'points': points}

        return sample


