import math
import os
import random

import cv2 as cv
import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from tf_creator import read
from utils import decode_segmap

data_transforms = {
    'train': transforms.Compose([
        transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
        # transforms.RandomApply([
        #     transforms.RandomOrder([
        #         transforms.RandomApply([transforms.RandomAffine(degrees=30, scale=[0.8, 1.25], shear=10)]),
        #         transforms.RandomApply([transforms.RandomHorizontalFlip(0.5)]), ])]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'trainval': transforms.Compose([
        transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
        # transforms.RandomApply([
        #     transforms.RandomOrder([
        #         transforms.RandomApply([transforms.RandomAffine(degrees=30, scale=[0.8, 1.25], shear=10)]),
        #         transforms.RandomApply([transforms.RandomHorizontalFlip(0.5)]), ])]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def safe_crop(image, mask, size=512):
    h, w = image.shape[0], image.shape[1]
    x = 0
    y = 0
    # x = np.random.randint(0, max(h - size, 0))
    # y = np.random.randint(0, max(w - size, 0))
    crop_image = image[x:x + size, y:y + size]
    crop_mask = mask[x:x + size, y:y + size]
    h, w = crop_image.shape[0], crop_image.shape[1]
    if h < size or w < size:
        x = np.random.randint(0, max(size - h, 0))
        y = np.random.randint(0, max(size - w, 0))
        placeholder_image = np.zeros((size, size, 3))
        placeholder_mask = np.zeros((size, size))
        placeholder_mask.fill(255)
        placeholder_image[x:x + h, y:y + w] = crop_image
        placeholder_mask[x:x + h, y:y + w] = crop_mask
        crop_image = placeholder_image
        crop_mask = placeholder_mask
    return crop_image, crop_mask


# images = read("trainval", "./data/pascal_voc_seg/tfrecord/")
# images = list(images)
# print(len(images))
#
# image = images[0]
# encoded = image['image/encoded']
# encoded = tf.io.decode_image(encoded, 3).numpy()
# print(encoded.shape)

class DLDataset(Dataset):
    def __init__(self, split, dataset_dir):
        self.split = split

        self.images = read(split, dataset_dir)
        self.images = list(self.images)
        # self.dataset_dir = dataset_dir
        # with open(dataset_dir + "ImageSets/Segmentation/" + split + ".txt") as file:
        #     self.names = file.read().splitlines()

        self.transformer = data_transforms[split]

    def __getitem__(self, i):
        item = self.images[i]

        image = item['image/encoded']
        image = tf.io.decode_image(image, 3)
        image = image.numpy()
        # image = Image.fromarray(image)
        # image = self.transformer(image)

        mask = item['image/segmentation/class/encoded']
        mask = tf.io.decode_image(mask, 1)
        mask = mask.numpy()
        mask = mask.reshape(mask.shape[:2])
        # mask = torch.from_numpy(mask)

        # filename = self.names[i]

        # image = Image.open(self.dataset_dir + "JPEGImages/" + filename + ".jpg")
        # mask = Image.open(self.dataset_dir + "SegmentationClassRaw/" + filename + ".png")
        # image = np.array(image)
        # mask = np.array(mask)



        image, mask = safe_crop(image, mask, size=512)
        image = transforms.ToPILImage()(image.copy().astype(np.uint8))
        image = self.transformer(image)
        return image, mask

    def __len__(self):
        return len(self.images)
        # return len(self.names)


if __name__ == "__main__":
    dltrain = DLDataset('val', "./data/pascal_voc_seg/tfrecord/")
    # dltrain = DLDataset('trainval', "./data/pascal_voc_seg/VOCdevkit/VOC2012/")
    dataloader = DataLoader(dltrain, batch_size=1, shuffle=True)
    for image, mask in dataloader:
        image = image.numpy()
        image = image[0]
        # print(image.shape)
        image = np.transpose(image, (1, 2, 0))
        plt.imshow(image)
        plt.show()

        mask = mask.numpy()
        mask = mask[0]
        print(type(mask))
        segmap = decode_segmap(mask, 'pascal', plot=True)
