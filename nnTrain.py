# Trai Neural Network
import torch
import torch.optim as optim
import torch.nn as nn

import LoadData
import nnModel
import nnTest

batch_size = 8
n_rows = None
train_loader, test_loader = LoadData.load_data(LoadData.path, nr_rows=n_rows, transform=LoadData.transform, batch_size=batch_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = True

net = nnModel.Net()
net = net.to(device)
criterion = nn.MultiLabelSoftMarginLoss()
learning_rate = 0.001
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
decay_rate = 0.85
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decay_rate)

for epoch in range(2):  # loop over the dataset multiple times

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
        # print(loss.dtype)
        # break
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 10 mini-batches
            print('[%d, %5d] loss: %.4f' %
                  (epoch + 1, (i + 1)*batch_size, running_loss / 100))
            running_loss = 0.0
    lr_scheduler.step()

print('Finished Training')
path = None     # Adauga path pentru salvare model
# torch.save(net.state_dict(), path)
nnTest.test_model(test_loader, net=net, path=path)
