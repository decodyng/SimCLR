from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10
import torch as th
import os

class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        save_image(img, 'results/img_pre_trans.png')
        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target

def save_image(image, file_path):
    """Save an RGB Torch tensor to a file. It is assumed that rgb_tensor is of
    shape [3,H,W] (channels-first), and that it has values in [0,1]."""

    dir_path = os.path.dirname(file_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    image.save(file_path)


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
