import torch.nn as nn
import torch

class Net(nn.Module):
    def __init__(self, out_shape=42):
        super(Net, self).__init__()
        # in shape (1,128,128)
        self.conv1 = nn.Conv2d(1, 5, 5, padding=2)
        self.conv2 = nn.Conv2d(5, 13, 9, padding=4)
        self.conv3 = nn.Conv2d(13, 17, 5, padding=2)
        self.conv4 = nn.Conv2d(17, 21, 13, padding=6)
        self.conv5 = nn.Conv2d(21, 25, 5, stride=2)
        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(25 * 15 * 15, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, out_shape)

        self.drop20 = nn.Dropout(0.2)
        self.drop30 = nn.Dropout(0.3)
        self.drop40 = nn.Dropout(0.4)
        self.drop2d = nn.Dropout2d(0.3)
        self.conv2_bn = nn.BatchNorm2d(17)
        self.fc2_bn = nn.BatchNorm1d(256)

    def forward(self, x):
        x = torch.tanh(self.drop2d(self.conv1(x)))
        x = self.pool(torch.tanh(self.conv2(x)))
        x = torch.tanh(self.conv2_bn(self.conv3(x)))
        x = torch.tanh(self.drop2d(self.conv4(x)))
        x = self.pool(torch.tanh(self.conv5(x)))
        # flatten
        x = x.view(-1, 25 * 15 * 15)
        x = torch.tanh(self.fc1(x))
        x = self.drop40(x)
        x = torch.tanh(self.fc2(x))
        x = self.drop20(x)
        x = torch.tanh(self.fc2_bn(self.fc3(x)))
        x = torch.tanh(self.out(x))
        return x

