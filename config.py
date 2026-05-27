import os
import torch


class Config:
    IMG_DIR = './data/image'
    MASK_DIR = './data/mask'

    OUTPUT_DIR = './experiment_results'
    RESULTS_DIR = './experiment_results/results'
    SEGMENTATION_RESULTS_DIR = './experiment_results/segmentation_predictions'
    YOLO_RESULTS_DIR = './experiment_results/yolo_results'
    CHECKPOINT_DIR = './experiment_results/checkpoints'

    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    RESIZE_SCHEDULE = [128, 192, 256, 320, 384]
    EPOCHS_PER_STAGE = [10, 10, 15, 15, 10]

    CONFIDENCE_THRESHOLD = 0.65

    YOLO_IMG_SIZE = 640
    YOLO_EPOCHS = 50
    YOLO_CONF_THRESHOLD = 0.25
    YOLO_MIN_AREA = 500

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    @classmethod
    def initialize_paths(cls):
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.RESULTS_DIR, exist_ok=True)
        os.makedirs(cls.SEGMENTATION_RESULTS_DIR, exist_ok=True)
        os.makedirs(cls.YOLO_RESULTS_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)


cfg = Config()