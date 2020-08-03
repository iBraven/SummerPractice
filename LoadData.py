from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from Dataset import HandsDataset
from torch.utils.data import DataLoader, random_split
from math import ceil
from nnModel import Net
nr_rows = 111
batch_size = 16
train_ratio = 0.9
transform = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.ColorJitter(contrast=1.6, brightness=0.2),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])])

hands_set = HandsDataset('c:\\Users\\rober\\PycharmProjects\\TorchTest\\test.csv',
                         rows=nr_rows,
                         transform=transform)


train_len = int(train_ratio * len(hands_set))
train_set, test_set = random_split(hands_set, [train_len, len(hands_set) - train_len])
print(len(test_set), len(train_set), len(hands_set))

data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
data_iter = iter(data_loader)


# print('data1', next(data_iter), '\ndata2:', next(data_iter))
def show_batch(sample_batched):
    images = sample_batched['image']
    for i in range(len(images)):
        plt.subplot(4, 4, i+1)
        plt.imshow(images[i].reshape(256, 256), cmap="gray")

# for batch_num in range(ceil(len(train_set)/batch_size)):
#     ax = plt.figure()
#     show_batch(next(data_iter))
#     plt.ioff()
#     plt.show()

net = Net()
# print(train_set[0]['image'])
predicted = net(train_set[0]['image'].reshape(-1, 1, 256, 256))
print(predicted)
print(len(predicted[0]))