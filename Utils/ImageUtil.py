import numpy as np
from scipy.ndimage import rotate
"""
ToRandomRotationAndCrop: 
"""
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