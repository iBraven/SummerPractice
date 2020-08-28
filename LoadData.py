from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from Dataset import HandsDataset
from torch.utils.data import DataLoader, random_split


rows = 10000
reshape_size = 128
batch = 16
train_test_ratio = 0.9
csv_path = 'c:\\Users\\rober\\PycharmProjects\\TorchTest\\points2D.csv'
img_transform = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.ColorJitter(contrast=1.6, brightness=0.2),
    transforms.Grayscale(),
    transforms.Resize(reshape_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])])


def load_data(path, nr_rows=None, batch_size=16, train_ratio=0.8, transform=None):
    """
    :param path:path for csv file
    :param nr_rows:get nr_rows from csv file
                    None = get all from file
    :param batch_size:size of batch for data loader
    :param train_ratio:ratio to split data into train and test
    :param transform:tranformation for image
    :return:Loader for train_set and test_set
    """
    hands_set = HandsDataset(path,
                             rows=nr_rows,
                             transform=transform)
    train_len = int(train_ratio * len(hands_set))
    train_set, test_set = random_split(hands_set, [train_len, len(hands_set) - train_len])
    print(len(test_set), len(train_set), len(hands_set))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def show_batch(sample_batched, net_data=None):
    images = sample_batched['image']
    points = sample_batched['points']
    size = (len(images)) if len(images) < 4 else 4
    points = denormalize_points(points, reshape_size)
    if net_data is not None:
        net_data = denormalize_points(net_data, reshape_size)
    for i in range(size):
        plt.subplot(2, 2, i+1)
        plt.imshow(images[i].reshape(reshape_size, reshape_size), cmap="gray")
        plt.scatter(points[i][::2], points[i][1::2], s=1.5, c='blue')
        if net_data is not None:
            plt.scatter(net_data[i][::2], net_data[i][1::2], s=0.9, c='red')

    plt.show()


def denormalize_points(points, out_shape=128):
    return (points + 1) * out_shape / 2


# train, test = load_data(csv_path, nr_rows=rows, transform=img_transform, batch_size=32)
# data = next(iter(train))
# show_batch(data)
#
# print(data['points'])
