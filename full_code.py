# ============================================================================
# INSTALL DEPENDENCIES
# ============================================================================
!pip install ultralytics segmentation-models-pytorch albumentations tqdm scikit-learn scipy seaborn openpyxl --quiet

# ============================================================================
# IMPORTS
# ============================================================================
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import warnings
import random
from scipy import ndimage
from ultralytics import YOLO
import shutil
from datetime import datetime
import json
import glob
warnings.filterwarnings('ignore')

try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# ============================================================================
# CONFIGURATION - MODIFIED FOR LOCAL SYSTEM (D:\data)
# ============================================================================
class Config:
    # Paths - Using local system path D:\data
    IMG_DIR = r"D:\data\image"  # Images folder
    MASK_DIR = r"D:\data\mask"  # Masks folder
    
    # Output directory for results
    OUTPUT_DIR = r"D:\data\experiment_results"
    RESULTS_DIR = r"D:\data\experiment_results\results"
    SEGMENTATION_RESULTS_DIR = r"D:\data\experiment_results\segmentation_predictions"
    YOLO_RESULTS_DIR = r"D:\data\experiment_results\yolo_results"
    CHECKPOINT_DIR = r"D:\data\experiment_results\checkpoints"

    # Training parameters
    EPOCHS_PER_SIZE = 1
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    # Progressive resizing schedule
    RESIZE_SCHEDULE = [128, 192, 256, 320, 384]
    EPOCHS_PER_STAGE = [10, 10, 15, 15, 10]

    # Resume training configuration
    RESUME_TRAINING = True
    SAVE_CHECKPOINTS = True

    # Loss weights
    DICE_WEIGHT = 0.5
    BCE_WEIGHT = 0.5
    BOUNDARY_WEIGHT = 0.4

    # Inference
    CONFIDENCE_THRESHOLD = 0.65
    MIN_REGION_AREA = 100

    # YOLO parameters
    YOLO_IMG_SIZE = 640
    YOLO_EPOCHS = 50
    YOLO_CONF_THRESHOLD = 0.25
    YOLO_MIN_AREA = 500

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    @classmethod
    def initialize_paths(cls):
        """Create necessary directories"""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.RESULTS_DIR, exist_ok=True)
        os.makedirs(cls.SEGMENTATION_RESULTS_DIR, exist_ok=True)
        os.makedirs(cls.YOLO_RESULTS_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        
        print(f"📁 Images path: {cls.IMG_DIR}")
        print(f"📁 Masks path: {cls.MASK_DIR}")
        print(f"📁 Results will be saved to: {cls.OUTPUT_DIR}")

# Create config instance
cfg = Config()

# ============================================================================
# DATASET CLASS
# ============================================================================
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

        # Resize
        image = cv2.resize(image, (self.target_size[1], self.target_size[0]))
        mask = cv2.resize(mask, (self.target_size[1], self.target_size[0]))

        # Augmentations
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

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================
class BoundaryAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode='binary')

    def forward(self, pred, target):
        dice_loss = self.dice(pred, target)

        # Boundary loss
        pred_prob = torch.sigmoid(pred)
        sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).to(pred.device)
        sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).to(pred.device)
        sobel_x = sobel_x.view(1,1,3,3)
        sobel_y = sobel_y.view(1,1,3,3)

        pred_edge = torch.sqrt(F.conv2d(pred_prob, sobel_x, padding=1)**2 +
                              F.conv2d(pred_prob, sobel_y, padding=1)**2 + 1e-8)
        target_edge = torch.sqrt(F.conv2d(target, sobel_x, padding=1)**2 +
                                F.conv2d(target, sobel_y, padding=1)**2 + 1e-8)

        boundary_dice = (2 * (pred_edge * target_edge).sum() + 1e-6) / \
                       (pred_edge.sum() + target_edge.sum() + 1e-6)

        return dice_loss + 0.4 * (1 - boundary_dice)

def get_loss(model_name):
    if model_name == 'unet_boundary':
        return BoundaryAwareLoss()
    else:
        return smp.losses.DiceLoss(mode='binary')

# ============================================================================
# MODEL FACTORY - ALL 10 MODELS
# ============================================================================
class ModelFactory:
    @staticmethod
    def create_model(model_name):
        models = {
            'unet': smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1),
            'unet_boundary': smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1, decoder_attention_type='scse'),
            'attention_unet': smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1, decoder_attention_type='scse'),
            'unet_plusplus': smp.UnetPlusPlus(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1),
            'deeplabv3': smp.DeepLabV3Plus(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1),
            'pspnet': smp.PSPNet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1),
            'fpn': smp.FPN(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1),
            'pan': smp.PAN(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1),
            'linknet': smp.Linknet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1),
            'manet': smp.MAnet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1)
        }
        return models[model_name]

# ============================================================================
# PROGRESSIVE TRAINER FOR SEGMENTATION MODELS - WITH RESUME CAPABILITY
# ============================================================================
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
        # Create model-specific directory for predictions
        self.pred_dir = os.path.join(cfg.SEGMENTATION_RESULTS_DIR, model_name)
        os.makedirs(self.pred_dir, exist_ok=True)

        # Checkpoint paths
        self.checkpoint_dir = os.path.join(cfg.CHECKPOINT_DIR, model_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(self.checkpoint_dir, 'training_state.pth')
        self.completed_stages_file = os.path.join(self.checkpoint_dir, 'completed_stages.json')

    def save_checkpoint(self, stage_idx, epoch, optimizer_state, best_dice):
        """Save training checkpoint"""
        if not cfg.SAVE_CHECKPOINTS:
            return

        checkpoint = {
            'stage_idx': stage_idx,
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer_state,
            'best_dice': best_dice,
            'model_name': self.model_name
        }
        torch.save(checkpoint, self.checkpoint_file)

        # Save completed stages info
        completed_info = {
            'completed_stages': stage_idx,
            'best_dice': best_dice,
            'last_updated': datetime.now().isoformat()
        }
        with open(self.completed_stages_file, 'w') as f:
            json.dump(completed_info, f, indent=2)

        print(f"    💾 Checkpoint saved (Stage {stage_idx+1}, Epoch {epoch+1})")

    def load_checkpoint(self):
        """Load training checkpoint if exists"""
        if not cfg.RESUME_TRAINING:
            return None, None, None, None

        if os.path.exists(self.checkpoint_file):
            print(f"    🔄 Found checkpoint for {self.model_name}, resuming training...")
            checkpoint = torch.load(self.checkpoint_file, map_location=cfg.DEVICE)

            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.best_dice = checkpoint['best_dice']

            stage_idx = checkpoint['stage_idx']
            start_epoch = checkpoint['epoch'] + 1
            optimizer_state = checkpoint['optimizer_state_dict']

            print(f"    ✓ Resuming from Stage {stage_idx+1}, Epoch {start_epoch}")
            print(f"    ✓ Previous best Dice: {self.best_dice:.4f}")

            return stage_idx, start_epoch, optimizer_state, checkpoint
        else:
            print(f"    ℹ️ No checkpoint found for {self.model_name}, starting fresh")
            return None, None, None, None

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(cfg.DEVICE), masks.to(cfg.DEVICE)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(images), masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        total_dice = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(cfg.DEVICE), masks.to(cfg.DEVICE)
                probs = torch.sigmoid(self.model(images))
                preds = (probs > cfg.CONFIDENCE_THRESHOLD).float()
                intersection = (preds * masks).sum()
                dice = (2 * intersection + 1e-6) / (preds.sum() + masks.sum() + 1e-6)
                total_dice += dice.item()
        return total_dice / len(val_loader)

    def train(self):
        print(f"\n▶ Training {self.model_name} with progressive resizing...")
        print(f"   Predictions will be saved to: {self.pred_dir}")

        # Try to resume from checkpoint
        start_stage, start_epoch_in_stage, saved_optimizer_state, checkpoint = self.load_checkpoint()

        # Track if we're resuming
        is_resuming = start_stage is not None

        for stage_idx, (size, epochs) in enumerate(zip(cfg.RESIZE_SCHEDULE, cfg.EPOCHS_PER_STAGE)):
            # Skip stages that are already completed
            if is_resuming and stage_idx < start_stage:
                print(f"  Stage {stage_idx+1}: {size}x{size} - SKIPPING (already completed)")
                continue

            # If this is the stage we're resuming from, adjust epochs
            if is_resuming and stage_idx == start_stage:
                stage_epochs = epochs - start_epoch_in_stage
                start_epoch_offset = start_epoch_in_stage
                print(f"  Stage {stage_idx+1}: {size}x{size} - RESUMING from epoch {start_epoch_in_stage+1}/{epochs}")
            else:
                stage_epochs = epochs
                start_epoch_offset = 0
                print(f"  Stage {stage_idx+1}: {size}x{size} ({epochs} epochs)")

            # Create datasets
            train_dataset = DesertWaterDataset(
                self.train_imgs, self.train_masks,
                target_size=(size, size), augment=True
            )
            val_dataset = DesertWaterDataset(
                self.val_imgs, self.val_masks,
                target_size=(size, size), augment=False
            )

            train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2)

            # Reinitialize optimizer for each stage (or load saved state if resuming)
            if is_resuming and stage_idx == start_stage and saved_optimizer_state is not None:
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.LEARNING_RATE)
                self.optimizer.load_state_dict(saved_optimizer_state)
            else:
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.LEARNING_RATE)

            # Training loop
            for epoch in range(stage_epochs):
                actual_epoch = start_epoch_offset + epoch
                train_loss = self.train_epoch(train_loader)
                val_dice = self.validate(val_loader)

                if val_dice > self.best_dice:
                    self.best_dice = val_dice
                    # Save best model checkpoint
                    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
                    torch.save(self.model.state_dict(), f"{cfg.OUTPUT_DIR}/{self.model_name}_best.pth")

                if (actual_epoch + 1) % 5 == 0 or actual_epoch == epochs - 1:
                    print(f"    Epoch {actual_epoch+1}/{epochs}: Val Dice={val_dice:.4f}, Loss={train_loss:.4f}")

                # Save checkpoint after each epoch
                self.save_checkpoint(stage_idx, actual_epoch, self.optimizer.state_dict(), self.best_dice)

            # After completing a stage, reset resume flags
            is_resuming = False
            start_stage = None

        print(f"  ✓ {self.model_name} best Dice: {self.best_dice:.4f}")

        # Clean up checkpoint files after successful training
        if os.path.exists(self.checkpoint_file):
            # Rename to completed
            completed_file = self.checkpoint_file.replace('.pth', '_completed.pth')
            shutil.move(self.checkpoint_file, completed_file)
            print(f"  ✓ Training completed! Checkpoint archived.")

        return self.best_dice

    def save_all_predictions(self, val_imgs, val_masks):
        """Save all segmentation predictions for this model"""
        print(f"  💾 Saving predictions for {self.model_name}...")

        # Load best model
        best_model_path = f"{cfg.OUTPUT_DIR}/{self.model_name}_best.pth"
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=cfg.DEVICE))
        else:
            print(f"    ⚠️ Best model not found, using current model state")

        self.model.eval()

        # Create subdirectories for this model's predictions
        masks_dir = os.path.join(self.pred_dir, 'predicted_masks')
        probs_dir = os.path.join(self.pred_dir, 'probabilities')
        overlay_dir = os.path.join(self.pred_dir, 'overlays')

        for d in [masks_dir, probs_dir, overlay_dir]:
            os.makedirs(d, exist_ok=True)

        # Save predictions
        with torch.no_grad():
            for idx, (img_path, mask_path) in enumerate(tqdm(zip(val_imgs, val_masks), total=len(val_imgs))):
                # Load and process image
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (384, 384))
                img_tensor = torch.from_numpy(img_resized / 255.0).permute(2, 0, 1).unsqueeze(0).to(cfg.DEVICE).float()

                # Get prediction
                prob = torch.sigmoid(self.model(img_tensor))
                pred_mask = (prob > cfg.CONFIDENCE_THRESHOLD).cpu().numpy().astype(np.uint8)[0, 0]
                prob_map = prob.cpu().numpy()[0, 0]

                # Save mask and probability map
                np.save(os.path.join(masks_dir, f'pred_mask_{idx}.npy'), pred_mask)
                np.save(os.path.join(probs_dir, f'prob_map_{idx}.npy'), prob_map)

                # Save as image
                cv2.imwrite(os.path.join(masks_dir, f'pred_mask_{idx}.png'), pred_mask * 255)

                # Create overlay visualization
                overlay = img_resized.copy()
                overlay[pred_mask == 1] = [0, 255, 0]  # Green for predictions
                overlay = cv2.addWeighted(img_resized, 0.6, overlay, 0.4, 0)
                cv2.imwrite(os.path.join(overlay_dir, f'overlay_{idx}.png'), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'num_predictions': len(val_imgs),
            'image_size': 384,
            'confidence_threshold': cfg.CONFIDENCE_THRESHOLD,
            'timestamp': datetime.now().isoformat()
        }
        with open(os.path.join(self.pred_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"  ✓ Saved {len(val_imgs)} predictions for {self.model_name}")

# ============================================================================
# EVALUATION METRICS
# ============================================================================
def compute_metrics(pred_mask, gt_mask):
    pred_mask = post_process(pred_mask)

    tp = np.sum((pred_mask == 1) & (gt_mask == 1))
    fp = np.sum((pred_mask == 1) & (gt_mask == 0))
    fn = np.sum((pred_mask == 0) & (gt_mask == 1))

    dice = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    # Boundary IoU
    kernel = np.ones((5, 5), dtype=np.uint8)
    pred_boundary = ndimage.binary_dilation(pred_mask, kernel) ^ ndimage.binary_erosion(pred_mask, kernel)
    gt_boundary = ndimage.binary_dilation(gt_mask, kernel) ^ ndimage.binary_erosion(gt_mask, kernel)
    boundary_iou = (np.sum(pred_boundary & gt_boundary) + 1e-6) / \
                   (np.sum(pred_boundary | gt_boundary) + 1e-6)

    return dice, iou, precision, recall, f1, boundary_iou

def post_process(mask, min_area=100):
    labeled, num = ndimage.label(mask)
    sizes = ndimage.sum(mask, labeled, range(1, num + 1))
    result = np.zeros_like(mask)
    for i, size in enumerate(sizes, 1):
        if size >= min_area:
            result[labeled == i] = 1
    return result

# ============================================================================
# YOLOv8 UTILITIES - MODIFIED TO SAVE RESULTS LOCALLY
# ============================================================================
def prepare_yolo_dataset(image_paths, mask_paths, output_dir):
    """Convert segmentation masks to YOLO format"""
    train_images_dir = os.path.join(output_dir, 'images', 'train')
    val_images_dir = os.path.join(output_dir, 'images', 'val')
    train_labels_dir = os.path.join(output_dir, 'labels', 'train')
    val_labels_dir = os.path.join(output_dir, 'labels', 'val')

    for dir_path in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
        os.makedirs(dir_path, exist_ok=True)

    split_idx = int(len(image_paths) * 0.8)
    train_pairs = list(zip(image_paths[:split_idx], mask_paths[:split_idx]))
    val_pairs = list(zip(image_paths[split_idx:], mask_paths[split_idx:]))

    def process_pairs(pairs, img_dir, label_dir, dataset_type):
        total_boxes = 0
        for idx, (img_path, mask_path) in enumerate(tqdm(pairs, desc=f"YOLO {dataset_type}")):
            img = cv2.imread(img_path)
            if img is None: continue

            mask = cv2.imread(mask_path, 0)
            if mask is None: continue

            img = cv2.resize(img, (cfg.YOLO_IMG_SIZE, cfg.YOLO_IMG_SIZE))
            cv2.imwrite(os.path.join(img_dir, f"image_{idx}.jpg"), img)

            # Extract bounding boxes from mask
            mask = cv2.resize(mask, (cfg.YOLO_IMG_SIZE, cfg.YOLO_IMG_SIZE))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            boxes = []
            for contour in contours:
                if cv2.contourArea(contour) >= cfg.YOLO_MIN_AREA:
                    x, y, w, h = cv2.boundingRect(contour)
                    x_center = (x + w/2) / cfg.YOLO_IMG_SIZE
                    y_center = (y + h/2) / cfg.YOLO_IMG_SIZE
                    width = w / cfg.YOLO_IMG_SIZE
                    height = h / cfg.YOLO_IMG_SIZE
                    boxes.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                    total_boxes += 1

            label_path = os.path.join(label_dir, f"image_{idx}.txt")
            with open(label_path, 'w') as f:
                f.write('\n'.join(boxes))

        return total_boxes

    train_boxes = process_pairs(train_pairs, train_images_dir, train_labels_dir, "TRAIN")
    val_boxes = process_pairs(val_pairs, val_images_dir, val_labels_dir, "VAL")

    yaml_content = f"""
path: {output_dir}
train: images/train
val: images/val
nc: 1
names: ['water']
"""
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        f.write(yaml_content)

    print(f"✅ YOLO dataset ready: {train_boxes} train boxes, {val_boxes} val boxes")
    return os.path.join(output_dir, 'data.yaml')

def train_yolo(data_yaml, output_dir):
    """Train YOLOv8 model and save results locally"""
    print("\n🚀 Training YOLOv8...")
    try:
        model = YOLO('yolov8n.pt')
        results = model.train(
            data=data_yaml,
            epochs=cfg.YOLO_EPOCHS,
            imgsz=cfg.YOLO_IMG_SIZE,
            batch=16,
            device=cfg.DEVICE,
            project=output_dir,
            name='yolo_detector',
            exist_ok=True,
            verbose=False
        )
        return YOLO(os.path.join(output_dir, 'yolo_detector', 'weights', 'best.pt'))
    except Exception as e:
        print(f"⚠️ YOLO training failed: {e}")
        return None

def save_yolo_detections(yolo_model, val_imgs, output_dir):
    """Save YOLO detection results"""
    print(f"  💾 Saving YOLO detection results...")
    yolo_results_dir = os.path.join(output_dir, 'detection_results')
    os.makedirs(yolo_results_dir, exist_ok=True)

    all_detections = []

    for idx, img_path in enumerate(tqdm(val_imgs, desc="Saving YOLO results")):
        img = cv2.imread(img_path)
        results = yolo_model(img_path, conf=cfg.YOLO_CONF_THRESHOLD)[0]

        # Save annotated image
        annotated = results.plot()
        cv2.imwrite(os.path.join(yolo_results_dir, f'detection_{idx}.jpg'), annotated)

        # Save detection data
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            for box, conf in zip(boxes, confs):
                all_detections.append({
                    'image_idx': idx,
                    'image_path': img_path,
                    'bbox': box.tolist(),
                    'confidence': float(conf)
                })

    # Save detections as JSON
    with open(os.path.join(yolo_results_dir, 'all_detections.json'), 'w') as f:
        json.dump(all_detections, f, indent=2)

    print(f"  ✓ Saved {len(all_detections)} detections from {len(val_imgs)} images")
    return all_detections

# ============================================================================
# VISUALIZATION - MODIFIED TO SAVE LOCALLY
# ============================================================================
def plot_results(results, output_dir):
    """Plot comparison results"""
    df = pd.DataFrame(results).T
    df = df.sort_values('dice_mean', ascending=False)
    models = list(df.index)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Dice scores
    ax = axes[0, 0]
    dice_means = [results[m]['dice_mean'] for m in models]
    dice_stds = [results[m]['dice_std'] for m in models]
    colors = ['gold' if m == 'unet_boundary' else 'steelblue' for m in models]
    ax.barh(models, dice_means, xerr=dice_stds, color=colors, capsize=5, edgecolor='black')
    ax.set_xlabel('Dice Score')
    ax.set_title('(a) Dice Score Comparison')
    ax.axvline(x=max(dice_means), color='red', linestyle='--', linewidth=2)

    # Boundary IoU
    ax = axes[0, 1]
    boundary_ious = [results[m]['boundary_iou_mean'] for m in models]
    ax.barh(models, boundary_ious, color=colors, edgecolor='black')
    ax.set_xlabel('Boundary IoU')
    ax.set_title('(b) Boundary IoU (Our Contribution)')

    # Precision-Recall
    ax = axes[1, 0]
    precisions = [results[m]['precision'] for m in models]
    recalls = [results[m]['recall'] for m in models]
    for i, model in enumerate(models):
        ax.scatter(precisions[i], recalls[i], s=150,
                  c='gold' if model == 'unet_boundary' else 'steelblue',
                  edgecolor='black', alpha=0.7)
        ax.annotate(model, (precisions[i], recalls[i]), fontsize=8, ha='center')
    ax.set_xlabel('Precision')
    ax.set_ylabel('Recall')
    ax.set_title('(c) Precision-Recall Trade-off')
    ax.grid(True, alpha=0.3)

    # F1 Score
    ax = axes[1, 1]
    f1_scores = [results[m]['f1_score'] for m in models]
    ax.barh(models, f1_scores, color=colors, edgecolor='black')
    ax.set_xlabel('F1 Score')
    ax.set_title('(d) F1 Score Comparison')

    plt.suptitle('Multi-Model Comparison for Desert Water Body Segmentation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_results.png'), dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# UTILITY FUNCTIONS FOR SAVING RESULTS
# ============================================================================
def save_experiment_summary(all_results, yolo_metrics, output_dir):
    """Save complete experiment summary"""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'resize_schedule': cfg.RESIZE_SCHEDULE,
            'epochs_per_stage': cfg.EPOCHS_PER_STAGE,
            'batch_size': cfg.BATCH_SIZE,
            'learning_rate': cfg.LEARNING_RATE,
            'confidence_threshold': cfg.CONFIDENCE_THRESHOLD,
            'device': cfg.DEVICE,
            'resume_training': cfg.RESUME_TRAINING
        },
        'segmentation_results': all_results,
        'yolo_results': yolo_metrics
    }

    with open(os.path.join(output_dir, 'experiment_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Also save as CSV for easy viewing
    df = pd.DataFrame(all_results).T
    df.to_csv(os.path.join(output_dir, 'segmentation_results.csv'))

def compress_results(output_dir):
    """Compress results for easy backup"""
    print("\n📦 Compressing results...")
    zip_path = shutil.make_archive(output_dir, 'zip', output_dir)
    print(f"✅ Results compressed to: {zip_path}")
    return zip_path

# ============================================================================
# MAIN FUNCTION - WITH RESUME CAPABILITY
# ============================================================================
def main():
    print("="*70)
    print("COMPLETE MULTI-MODEL COMPARISON + YOLOv8")
    print("Desert Water Body Segmentation in Hyper-Arid Regions")
    print("SAVING RESULTS LOCALLY")
    print("="*70)

    # Initialize configuration paths
    cfg.initialize_paths()

    # Print resume status
    if cfg.RESUME_TRAINING:
        print("🔄 RESUME MODE ENABLED - Training will continue from last checkpoint")
    else:
        print("✨ FRESH START MODE - Training from scratch")

    # Load data
    valid_extensions = ('.png', '.jpg', '.jpeg')
    images = sorted([os.path.join(cfg.IMG_DIR, f) for f in os.listdir(cfg.IMG_DIR)
                    if f.lower().endswith(valid_extensions)])
    masks = sorted([os.path.join(cfg.MASK_DIR, f) for f in os.listdir(cfg.MASK_DIR)
                   if f.lower().endswith(valid_extensions)])

    print(f"\nFound {len(images)} images, {len(masks)} masks")

    if len(images) == 0:
        print("❌ No images found! Please check your paths.")
        print("Expected paths:")
        print(f"  Images: {cfg.IMG_DIR}")
        print(f"  Masks: {cfg.MASK_DIR}")
        return

    # Split data
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        images, masks, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE
    )

    print(f"Training: {len(train_imgs)} samples")
    print(f"Validation: {len(val_imgs)} samples")
    print(f"Device: {cfg.DEVICE}")
    print(f"Progressive schedule: {cfg.RESIZE_SCHEDULE}")

    # ========================================================================
    # PART 1: Train all 10 segmentation models with progressive resizing
    # ========================================================================
    print("\n" + "="*60)
    print("PART 1: TRAINING 10 SEGMENTATION MODELS")
    print("="*60)

    model_names = ['unet', 'unet_boundary', 'attention_unet', 'unet_plusplus',
                   'deeplabv3', 'pspnet', 'fpn', 'pan', 'linknet', 'manet']

    all_results = {}
    trainers = []  # Store trainers to save predictions later

    for model_name in model_names:
        model = ModelFactory.create_model(model_name)
        trainer = ProgressiveTrainer(model, model_name, train_imgs, train_masks, val_imgs, val_masks)

        # Check if this model is already fully trained
        best_model_path = f"{cfg.OUTPUT_DIR}/{model_name}_best.pth"
        if cfg.RESUME_TRAINING and os.path.exists(best_model_path):
            # Check if checkpoint is marked as completed
            completed_checkpoint = os.path.join(cfg.CHECKPOINT_DIR, model_name, 'training_state_completed.pth')
            if os.path.exists(completed_checkpoint):
                print(f"\n✅ {model_name} already fully trained. Loading model...")
                model.load_state_dict(torch.load(best_model_path, map_location=cfg.DEVICE))
                model.eval()

                # Load saved metrics if available
                metrics_file = os.path.join(cfg.CHECKPOINT_DIR, model_name, 'final_metrics.json')
                if os.path.exists(metrics_file):
                    with open(metrics_file, 'r') as f:
                        model_results = json.load(f)
                        all_results[model_name] = model_results
                        print(f"  ✓ Loaded saved metrics: Dice={model_results['dice_mean']:.4f}")
                        trainers.append(trainer)
                        continue

        # Train or resume training
        trainer.train()
        trainers.append(trainer)

        # Evaluate best model
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=cfg.DEVICE))
        model.eval()

        # Final evaluation at 384x384
        val_dataset = DesertWaterDataset(val_imgs, val_masks, target_size=(384, 384), augment=False)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        all_metrics = []
        with torch.no_grad():
            for img, mask in tqdm(val_loader, desc=f"Evaluating {model_name}"):
                img = img.to(cfg.DEVICE).float()
                prob = torch.sigmoid(model(img))
                pred = (prob > cfg.CONFIDENCE_THRESHOLD).cpu().numpy().astype(np.uint8)[0, 0]
                gt = mask.cpu().numpy().astype(np.uint8)[0, 0]
                all_metrics.append(compute_metrics(pred, gt))

        all_results[model_name] = {
            'dice_mean': np.mean([m[0] for m in all_metrics]),
            'dice_std': np.std([m[0] for m in all_metrics]),
            'iou_mean': np.mean([m[1] for m in all_metrics]),
            'boundary_iou_mean': np.mean([m[5] for m in all_metrics]),
            'precision': np.mean([m[2] for m in all_metrics]),
            'recall': np.mean([m[3] for m in all_metrics]),
            'f1_score': np.mean([m[4] for m in all_metrics]),
        }

        # Save metrics for this model
        metrics_file = os.path.join(cfg.CHECKPOINT_DIR, model_name, 'final_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(all_results[model_name], f, indent=2)

        print(f"  ✓ {model_name}: Dice={all_results[model_name]['dice_mean']:.4f}")

    # Save all segmentation predictions
    print("\n" + "="*60)
    print("SAVING SEGMENTATION PREDICTIONS FOR ALL MODELS")
    print("="*60)

    for trainer in trainers:
        trainer.save_all_predictions(val_imgs, val_masks)

    # ========================================================================
    # PART 2: YOLOv8 Object Detection
    # ========================================================================
    print("\n" + "="*60)
    print("PART 2: YOLOv8 OBJECT DETECTION")
    print("="*60)

    # Prepare YOLO dataset locally
    yolo_output_dir = os.path.join(cfg.YOLO_RESULTS_DIR, 'yolo_dataset')
    data_yaml = prepare_yolo_dataset(images, masks, yolo_output_dir)

    # Train YOLO
    yolo_model = train_yolo(data_yaml, cfg.YOLO_RESULTS_DIR)

    yolo_metrics = {'total_detections': 0, 'detection_rate': 0}
    if yolo_model:
        # Save YOLO detection results
        all_detections = save_yolo_detections(yolo_model, val_imgs, cfg.YOLO_RESULTS_DIR)

        yolo_metrics = {
            'total_detections': len(all_detections),
            'detection_rate': len(all_detections) / len(val_imgs) if len(val_imgs) > 0 else 0
        }
        print(f"✅ YOLO: {yolo_metrics['total_detections']} detections on {len(val_imgs)} images")
        print(f"   Detection rate: {yolo_metrics['detection_rate']:.2f} objects/image")

    # ========================================================================
    # PART 3: FINAL RESULTS
    # ========================================================================
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)

    # Save experiment summary
    save_experiment_summary(all_results, yolo_metrics, cfg.RESULTS_DIR)

    # Create comparison table
    df = pd.DataFrame(all_results).T
    df = df.sort_values('dice_mean', ascending=False)

    print("\n📊 Model Comparison:")
    print(df[['dice_mean', 'dice_std', 'iou_mean', 'boundary_iou_mean', 'precision', 'recall', 'f1_score']].round(4))

    # Best model
    best_model = df.index[0]
    our_results = all_results.get('unet_boundary', None)

    print(f"\n🏆 BEST MODEL: {best_model}")

    if our_results:
        our_rank = list(all_results.keys()).index('unet_boundary') + 1
        print(f"\n⭐ OUR METHOD (unet_boundary) Rank: #{our_rank} out of {len(all_results)}")
        print(f"   Dice: {our_results['dice_mean']:.4f} ± {our_results['dice_std']:.4f}")
        print(f"   Boundary IoU: {our_results['boundary_iou_mean']:.4f}")
        print(f"   Precision: {our_results['precision']:.4f}")
        print(f"   Recall: {our_results['recall']:.4f}")

    print(f"\n🎯 YOLOv8: {yolo_metrics['detection_rate']:.2f} objects/image")

    # Plot and save
    plot_results(all_results, cfg.RESULTS_DIR)

    # ========================================================================
    # SAVE AND COMPRESS RESULTS
    # ========================================================================
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)

    print(f"\n📁 Results saved at: {cfg.OUTPUT_DIR}")
    print("\nFolder structure:")
    print(f"  {cfg.OUTPUT_DIR}/")
    print(f"    ├── results/               # Metrics and summaries")
    print(f"    ├── segmentation_predictions/  # All 10 models' predictions")
    print(f"    │   ├── unet/")
    print(f"    │   ├── unet_boundary/")
    print(f"    │   └── ...")
    print(f"    ├── checkpoints/           # Training checkpoints for resume")
    print(f"    └── yolo_results/          # YOLO detection results")

    # Optionally compress results
    compress_choice = input("\nDo you want to compress all results? (yes/no): ").lower()
    if compress_choice == 'yes':
        zip_path = compress_results(cfg.OUTPUT_DIR)
        print(f"✅ Results compressed to: {zip_path}")

    print("\n" + "="*60)
    print("✅ EXPERIMENT COMPLETE - READY FOR Q1 PAPER")
    print(f"📁 All results saved in: {cfg.OUTPUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()