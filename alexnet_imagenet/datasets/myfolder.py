import torch.utils.data as data

from PIL import Image
import os
import os.path


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def make_dataset_filetable(root, mapfile):
    imgs = []
    with open(mapfile, 'r') as f:
        records = f.readlines()
        for record in records:
            path, label = record.split(" ")
            path = os.path.join(root, path)
            label = int(label.replace(" ", ""))
            imgs.append((path, label))
    return imgs


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, target_transform=None, loader=pil_loader, mapfile=""):

        imgs = make_dataset_filetable(root, mapfile)

        self.root = root
        self.imgs = imgs
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imgs)
