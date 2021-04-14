from torch.utils.data import DataLoader
import os
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
])
to_tensor = transforms.Compose([
    transforms.ToTensor()
])



# Simple Image Loader --> For Test! No Ground Truth
# no augmentation, downsample (for too big image --> memory problem)
###############################################################################
class ImageLodader(DataLoader):
    def __init__(self, path):
        self.input_path = path

        self.input_list = os.listdir(self.input_path)


    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        input = Image.open(os.path.join(self.input_path, self.input_list[idx]))
        if input.width<1024 and input.height<1024:
            input = input.resize((int(input.width //32 *32), int(input.height//32 *32)))
        else:
            input = input.resize((int(input.width // 64 * 32), int(input.height // 64 * 32)))
        input = np.array(input)[..., :3]
        input = transform(input)

        return input

def load_singular_image(path, name):
    input = Image.open(os.path.join(path, name))
    input = input.resize((int(512), int(512)))
    input = np.array(input)[..., :3]
    input = transform(input).unsqueeze(0)
    return input