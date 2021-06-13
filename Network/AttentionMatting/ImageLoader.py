import os
import torchvision.transforms as transforms
import cv2
from torch.utils.data import DataLoader
"""
# Transform 
# ToTensor & Normalization(Only Input)
"""
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
transform_1c = transforms.Compose([
    transforms.ToTensor(),
])

"""
Simple Image Loader
Only Reshape --> Used Overfitting Test
getitem parameter: index & shape of image
"""
class ImageLoader(DataLoader):
    def __init__(self, path, gt_path):
        self.path = path
        self.gt_path = gt_path
        self.img_list = os.listdir(path)
        self.gt_list = os.listdir(gt_path)
    def __len__(self):
        return len(self.img_list)
    def __getitem__(self, index, shape=[256, 256]):
        img = cv2.imread(os.path.join(self.path, self.img_list[index]), cv2.IMREAD_COLOR)
        img = cv2.resize(img, dsize=shape, interpolation=cv2.INTER_AREA)
        gt = cv2.imread(os.path.join(self.gt_path, self.gt_list[index]), cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt, dsize=shape, interpolation=cv2.INTER_AREA)

        img = transform(img)
        gt = transform_1c(gt)
        return img, gt

