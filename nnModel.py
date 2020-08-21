import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, out_shape=42):
        super(Net, self).__init__()
        # in shape (1,128,128)
        self.conv1 = nn.Conv2d(1, 8, 5)
        # in shape (8,124,124)
        self.pool = nn.MaxPool2d(2, 2)
        # in shape (8,62,62)
        self.conv2 = nn.Conv2d(8, 16, 5)
        # in shape (16,58,58)
        # urmeaza pool -> (16,29,29)
        # in shape (16,29,29)
        self.conv3 = nn.Conv2d(16, 24, 5)

        # in shape (24,25,25)
        # layer 4608 nodes -> 1024 nodes
        self.fc1 = nn.Linear(24 * 25 * 25, 1024)
        # layer 1024 nodes -> 1024 nodes
        self.fc2 = nn.Linear(1024, 1024)
        # layer 1024 nodes -> 256 nodes
        self.fc3 = nn.Linear(1024, 256)
        # layer 256 nodes -> 63 nodes
        self.out = nn.Linear(256, out_shape)
        self.drop30 = nn.Dropout(0.3)
        self.drop20 = nn.Dropout(0.2)
        self.drop10 = nn.Dropout(0.1)
        self.conv2_bn = nn.BatchNorm2d(16)
        self.fc2_bn = nn.BatchNorm1d(256)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        # print(f'conv3: {x.shape}')
        # flatten
        x = x.view(-1, 24 * 25 * 25)
        x = F.relu(self.fc1(x))
        x = self.drop30(x)
        x = F.relu(self.fc2(x))
        x = self.drop10(x)
        x = F.relu(self.fc2_bn(self.fc3(x)))
        x = self.drop10(x)
        x = self.out(x)
        return x
#
# net = Net()
# print(net)
