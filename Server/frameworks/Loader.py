from torch.utils.data import DataLoader
import os
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import random
import torch.nn as nn
from scipy.ndimage import rotate


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
])

to_tensor = transforms.Compose([
    transforms.ToTensor()
])

# Random Crop & Rotation... without black area
# calculate profer range & rotate + crop
#################################################################
class ToRandomRotationAndCrop:
    def __init__(self, max_angle, new_size):
        self.max_angle = max_angle
        self.new_size = new_size
        self.on_rotation = 0
        self.top_left = None

    def set(self):
        self.on_rotation = ((np.random.rand() * 2)-1.0) * self.max_angle
        self.top_left = None

    def get_valid_indies(self, size):
        new_h, new_w = self.new_size

        mask = rotate(np.ones(size), self.on_rotation)
        mask[mask < 0.95] = 0
        mask[mask > 0] = 1

        mask_sum = np.sum(mask, axis=0)
        valid_axis_x = np.where(mask_sum >= new_h)[0]

        mask_sum = np.sum(mask, axis=1)
        valid_axis_y = np.where(mask_sum >= new_w)[0]

        candidate = []
        for x in valid_axis_x[:-new_w + 1]:
            for y in valid_axis_y[:-new_h + 1]:
                if mask[y, x] and mask[y, x + new_w - 1] and mask[y + new_h - 1, x] and mask[
                    y + new_h - 1, x + new_w - 1]:
                    candidate.append((y, x))
        return candidate[np.random.randint(0,len(candidate))]

    def apply(self, sample):
        sample = rotate(sample, self.on_rotation)
        return sample

# flip augmentation
def horizontal_flip(tensor):
    return torch.flip(tensor, [0, 2])

# flip augmentation
def vertical_flip(tensor):
    return torch.flip(tensor, [0, 1])

# TorchLoader: DataLoader for Training
# Load RGB Image & Ground Truth // Augmentation (Crop, Rotate, Flip)
####################################################################
class TorchLodader(DataLoader):
    def __init__(self, data_config, validation = False, aug=False):
        self.input_type = data_config['INPUT_TYPE']
        self.sub_input = data_config['SUB_INPUT']
        if not validation:
            self.input_path = data_config['INPUT_PATH']
            self.gt_path = data_config['GT_PATH']
        else:
            self.input_path = data_config['Valid_Path']
            self.gt_path = data_config['GT_valid']

        self.hint_path = data_config['HINT_PATH']

        self.input_w = data_config['INPUT_W']
        self.input_h = data_config['INPUT_H']

        self.crop_shape = data_config['CropShape']

        self.augmentation = aug
        self.reshaper = nn.Upsample(self.crop_shape, align_corners=False)

        self.input_list = os.listdir(self.input_path)
        self.gt_list = os.listdir(self.gt_path)
        self.hint_list = os.listdir(self.hint_path) if self.sub_input != 0 else None


    def __len__(self):
        return len(self.input_list)

    def safe_crop(self, img, gt, size):
        h, w, c = img.shape
        # resize
        if h<=size[0] or w<=size[1]:
            img = img.resize((512, 512))
            gt = gt.resize((512, 512))
            img = np.array(img)[..., :3]
            gt = np.array(gt)
        else:
            img = np.array(img)[..., :3]
            gt = np.array(gt)
            startPoint_x = random.randrange(0,w-size[1])
            startPoint_y = random.randrange(0, h-size[0])
            img = img[startPoint_y:startPoint_y+size[0], startPoint_x:startPoint_x+size[1]]
            gt = gt[startPoint_y:startPoint_y + size[0], startPoint_x:startPoint_x + size[1]]
        self.noise_intensity=0.005
        return img, gt

    def __getitem__(self, idx):
        input = Image.open(os.path.join(self.input_path, self.input_list[idx]))
        gt = Image.open(os.path.join(self.gt_path, self.gt_list[idx])).convert('L')

        if self.augmentation:
            if input.size[0] > self.crop_shape[0] and input.size[1] > self.crop_shape[1]:
                input = np.array(input)
                gt = np.array(gt)
                input, gt = self.safe_crop(input, gt, self.crop_shape)
            else:
                input = input.resize(self.crop_shape)
                gt = gt.resize(self.crop_shape)
        input = transform(input)
        gt = to_tensor(gt)
        if self.augmentation:
            flip_h = random.randint(0, 1)
            flip_w = random.randint(0, 1)
            input, gt = augmentation_tensor(input, gt, flip_h, flip_w)
        else:
            reshaper = nn.Upsample([512, 512])
            input = reshaper(input.unsqueeze(0)).squeeze(0)
            gt = reshaper(gt.unsqueeze(0)).squeeze(0)
        return input, gt

# Augmentation 'Tensor' (After Transform)
# flip Process
#######################################################################
def augmentation_tensor(input, gt, flip_h, flip_w):
    if flip_h == 1:
        input = horizontal_flip(input)
        gt = horizontal_flip(gt)
    if flip_w == 1:
        input = vertical_flip(input)
        gt = vertical_flip(gt)
    return input, gt