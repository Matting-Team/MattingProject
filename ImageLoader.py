import os
import torchvision.transforms as transforms
import torch
import PIL.Image as Image
import numpy as np
import cv2
from torch.utils.data import DataLoader
import PIL.Image as Image

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
transform_1c = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.5,), (0.5,))
])
class ImageLoader(DataLoader):
    def __init__(self, path, gt_path):
        self.path = path
        self.gt_path = gt_path
        self.img_list = os.listdir(path)
        self.gt_list = os.listdir(gt_path)
    def __len__(self):
        return len(self.img_list)
    def __getitem__(self, index):
        img = cv2.imread(os.path.join(self.path, self.img_list[index]), cv2.IMREAD_COLOR)
        img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_AREA)
        gt = cv2.imread(os.path.join(self.gt_path, self.gt_list[index]), cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt, dsize=(256, 256), interpolation=cv2.INTER_AREA)

        img = transform(img)
        gt = transform_1c(gt)
        return img, gt

