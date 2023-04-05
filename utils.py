import torch
import torchvision
from torchvision import transforms
from PIL import Image
import random

def custom_collate_fn(batch):
    """
    Custom collate function to handle images of different sizes.
    :param batch: list of tuples (image, label).
    """
    imgs = []
    for sample in batch:
        img, target = sample
        imgs.append(img)
    targets = torch.LongTensor([item[1] for item in batch])  # image labels.
    return imgs, targets



class CustomMNIST(torchvision.datasets.MNIST):
    """
    Custom MNIST dataset with random width/height resize.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='L')
        img = self.random_resize(img) # resize image to a random size
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def random_resize(self, img):
        new_size_w = random.randint(28, 56) # generate a random size between 28 and 56
        new_size_h = random.randint(28, 56)
        img = transforms.Resize((new_size_h,new_size_w))(img)
        return img