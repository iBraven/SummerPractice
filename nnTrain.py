#Trai Neural Network
import torch
import torch.optim as optim
import torch.nn as nn

import LoadData
import nnModel

train_loader, test_loader = LoadData.load_data(LoadData.path, nr_rows=100, transform=LoadData.transform)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.backends.cudnn.enabled=False
# torch.backends.cudnn.benchmark=True

net = nnModel.Net()
net = net.to(device)
criterion = nn.SmoothL1Loss()
optimizer = optim.SGD(net.parameters(), lr=0.001)

for epoch in range(1):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        image, points = data['image'].to(device), data['points'].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(image)
        # print(outputs.shape, points.shape, image.shape)
        # print(type(outputs), type(points), type(image))
        # break
        loss = criterion(outputs, points)
        print(loss.dtype)
        # break
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # print every 10 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

print('Finished Training')