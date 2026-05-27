import os
import json
import shutil
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from config import cfg
from dataset import DesertWaterDataset
from losses import get_loss


class ProgressiveTrainer:
    def __init__(self, model, model_name, train_imgs, train_masks, val_imgs, val_masks):
        self.model = model.to(cfg.DEVICE)
        self.model_name = model_name
        self.train_imgs = train_imgs
        self.train_masks = train_masks
        self.val_imgs = val_imgs
        self.val_masks = val_masks

        self.criterion = get_loss(model_name)
        self.best_dice = 0

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0

        for images, masks in train_loader:
            images = images.to(cfg.DEVICE)
            masks = masks.to(cfg.DEVICE)

            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        total_dice = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(cfg.DEVICE)
                masks = masks.to(cfg.DEVICE)

                probs = torch.sigmoid(self.model(images))
                preds = (probs > cfg.CONFIDENCE_THRESHOLD).float()

                intersection = (preds * masks).sum()
                dice = (
                    2 * intersection + 1e-6
                ) / (
                    preds.sum() + masks.sum() + 1e-6
                )

                total_dice += dice.item()

        return total_dice / len(val_loader)

    def train(self):
        print(f'Training {self.model_name}')

        for size, epochs in zip(cfg.RESIZE_SCHEDULE, cfg.EPOCHS_PER_STAGE):
            train_dataset = DesertWaterDataset(
                self.train_imgs,
                self.train_masks,
                target_size=(size, size),
                augment=True
            )

            val_dataset = DesertWaterDataset(
                self.val_imgs,
                self.val_masks,
                target_size=(size, size),
        print(f'Saving predictions for {self.model_name}')