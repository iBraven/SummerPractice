#Load Data
import torch
import torchvision as tv

image_path = "cale\\catre\\folder\\img"
image_size = {"width": 256, "height": 256}


class ImageLoader:
    def __init__(self, batch_size=32):
        self.dataset = tv.datasets.ImageFolder(image_path, transform=None)

    def return_data(self):
        return self.dataset
