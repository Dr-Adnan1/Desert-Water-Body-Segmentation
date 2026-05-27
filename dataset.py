import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class DesertWaterDataset(Dataset):
    def __init__(self, image_paths, mask_paths, target_size, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.target_size = target_size
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        image = cv2.resize(image, (self.target_size[1], self.target_size[0]))
        mask = cv2.resize(mask, (self.target_size[1], self.target_size[0]))

        if self.augment:
            if np.random.random() > 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()

            if np.random.random() > 0.5:
                image = np.flipud(image).copy()
                mask = np.flipud(mask).copy()

        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask