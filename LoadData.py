from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from Dataset import HandsDataset
from torch.utils.data import DataLoader

hands_set = HandsDataset('c:\\Users\\rober\\PycharmProjects\\TorchTest\\test.csv',
                         rows=100,
                         transform=transforms.Compose([
    transforms.ToPILImage(),
    # transforms.ColorJitter(contrast=1.6, brightness=0.2),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
                         ]))


data_loader = DataLoader(hands_set, batch_size=16, shuffle=True)
data_iter = iter(data_loader)
def show_batch(sample_batched):
    images = sample_batched['image']
    for i in range(len(images)):
        plt.subplot(4, int(len(images)/4), i+1)
        plt.imshow(images[i].reshape(256, 256))
        print(type(images[i]))

plt.figure()
show_batch(next(data_iter))
plt.ioff()
plt.show()
