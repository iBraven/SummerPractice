from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from Dataset import HandsDataset

hands_set = HandsDataset('c:\\Users\\rober\\PycharmProjects\\TorchTest\\test.csv',
                         rows=10,
                         transform=transforms.Compose([
    transforms.ColorJitter(brightness=0.4),
    transforms.Grayscale()])
                         )

for i in range(5):
    sample = hands_set[i]

    ax = plt.subplot(1, 5, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    plt.imshow(sample['image'])


plt.show()
