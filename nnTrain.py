#Trai Neural Network
import torch
import torch.optim as optim
import torch.nn as nn

import LoadData
import nnModel
import nnTest

batch_size = 16
n_rows = 1000
learning_rate = 0.005
decay_rate = 0.9
train_loader, test_loader = LoadData.load_data(LoadData.csv_path,
                                               nr_rows=n_rows,
                                               transform=LoadData.img_transform,
                                               batch_size=batch_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = True

net = nnModel.Net()
net = net.to(device)
criterion = nn.MultiLabelSoftMarginLoss()
# Optimizer and exponential learning rate
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decay_rate)

for epoch in range(15):  # loop over the dataset multiple times

    running_loss = 0.0
    print(f'epoch: {epoch + 1}, learning rate:', end=' ')
    for param_group in optimizer.param_groups:
        print(param_group['lr'])
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        image, points = data['image'].to(device), data['points'].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(image)
        loss = criterion(outputs, points)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 batches
            print('[%d, %6d] loss: %.4f' %
                  (epoch + 1, (i + 1)*batch_size, running_loss / 100))
            running_loss = 0.0
    # update learning rate
    lr_scheduler.step()

print('Finished Training')
# Save model
path = 'D:\\Model\\Model1.pth'     # Adauga path pentru salvare model
torch.save(net.state_dict(), path)
print('Test set')
nnTest.test_model(train_loader, net=net)
