import os
import cv2
import json
from tqdm import tqdm
from ultralytics import YOLO

from config import cfg


def prepare_yolo_dataset(image_paths, mask_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    yaml_content = f'''
path: {output_dir}
train: images/train
val: images/val
nc: 1
names: ['water']
'''

    yaml_path = os.path.join(output_dir, 'data.yaml')

    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    return yaml_path


def train_yolo(data_yaml, output_dir):
    model = YOLO('yolov8n.pt')

    model.train(
        data=data_yaml,
        epochs=cfg.YOLO_EPOCHS,
        imgsz=cfg.YOLO_IMG_SIZE,
        project=output_dir,
        name='yolo_detector'
    )

    return YOLO(os.path.join(output_dir, 'yolo_detector', 'weights', 'best.pt'))


def save_yolo_detections(yolo_model, val_imgs, output_dir):
    detections = []

    for img_path in tqdm(val_imgs):
        results = yolo_model(img_path, conf=cfg.YOLO_CONF_THRESHOLD)[0]

        detections.append({
            'image': img_path
        })

    with open(os.path.join(output_dir, 'detections.json'), 'w') as f:
        json.dump(detections, f, indent=2)

    return detections