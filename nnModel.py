import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # in shape (1,256,256)
        self.conv1 = nn.Conv2d(1, 8, 5)
        # in shape (8,252,252)
        self.pool = nn.MaxPool2d(2, 2)
        # in shape (8,126,126)
        self.conv2 = nn.Conv2d(8, 16, 5)
        # in shape (16,122,122)
        # urmeaza pool ->
        # in shape (16,61,61)
        self.conv3 = nn.Conv2d(16, 24, 6)
        # in shape (24,56,56)
        # urmeaza pool ->
        # in shape (24,28,28)
        self.conv4 = nn.Conv2d(24, 32, 5)
        # in shape (32,24,24)
        # urmeaza pool ->

        # in shape (32,12,12)
        # layer 25 * 56 * 56 nodes -> 2^14 nodes
        self.fc1 = nn.Linear(25 * 56 * 56, 16384)
        # layer 2^14 nodes -> 2^10 nodes
        self.fc2 = nn.Linear(16384, 1024)
        # layer 1024 nodes -> 256 nodes
        self.fc3 = nn.Linear(1024, 256)
        # layer 256 nodes -> 63 nodes
        self.out = nn.Linear(256, 63)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        # x = self.pool(F.relu(self.conv4(x)))
        # flatten
        x = x.view(-1, 25 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
#
# net = Net()
# print(net)
