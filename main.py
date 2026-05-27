from config import cfg
from models import ModelFactory
from train import ProgressiveTrainer
from dataset import DesertWaterDataset
from evaluate import compute_metrics
from visualization import plot_results
from yolo_utils import prepare_yolo_dataset, train_yolo, save_yolo_detections
from utils import save_experiment_summary, compress_results

import os
import torch
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


def main():
    print('='*70)
    print('COMPLETE MULTI-MODEL COMPARISON + YOLOv8')
    print('='*70)

    cfg.initialize_paths()

    valid_extensions = ('.png', '.jpg', '.jpeg')

    images = sorted([
        os.path.join(cfg.IMG_DIR, f)
        for f in os.listdir(cfg.IMG_DIR)
        if f.lower().endswith(valid_extensions)
    ])

    masks = sorted([
        os.path.join(cfg.MASK_DIR, f)
        for f in os.listdir(cfg.MASK_DIR)
        if f.lower().endswith(valid_extensions)
    ])

    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        images,
        masks,
        test_size=cfg.TEST_SIZE,
        random_state=cfg.RANDOM_STATE
    )

    model_names = [
        'unet', 'unet_boundary', 'attention_unet',
        'unet_plusplus', 'deeplabv3', 'pspnet',
        'fpn', 'pan', 'linknet', 'manet'
    ]

    all_results = {}
    trainers = []

    for model_name in model_names:
        model = ModelFactory.create_model(model_name)

        trainer = ProgressiveTrainer(
            model,
            model_name,
            train_imgs,
            train_masks,
            val_imgs,
            val_masks
        )

        trainer.train()
        trainers.append(trainer)

        best_model_path = f'{cfg.OUTPUT_DIR}/{model_name}_best.pth'

        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=cfg.DEVICE))

        model.eval()

        val_dataset = DesertWaterDataset(
            val_imgs,
            val_masks,
            target_size=(384, 384),
            augment=False
        )
    main()