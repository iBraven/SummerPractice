from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from Dataset import HandsDataset
from torch.utils.data import DataLoader, random_split


# nr_rows = 111
# batch_size = 16
# train_ratio = 0.9
path = 'c:\\Users\\rober\\PycharmProjects\\TorchTest\\test.csv'
transform = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.ColorJitter(contrast=1.6, brightness=0.2),
    transforms.Grayscale(),
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])])

def load_data(path, nr_rows = None, batch_size = 16, train_ratio = 0.8, transform = None):
    hands_set = HandsDataset(path,
                             rows=nr_rows,
                             transform=transform)

    train_len = int(train_ratio * len(hands_set))
    train_set, test_set = random_split(hands_set, [train_len, len(hands_set) - train_len])
    print(len(test_set), len(train_set), len(hands_set))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def show_batch(sample_batched):
    images = sample_batched['image']
    for i in range(len(images)):
        plt.subplot(8, 4, i+1)
        plt.imshow(images[i].reshape(128, 128), cmap="gray")

# train_loader, test_loader = load_data(path, nr_rows=None, transform=transform, batch_size=32)
# data = iter(train_loader)
# show_batch(next(data))
# plt.show()