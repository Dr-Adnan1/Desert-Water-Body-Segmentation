# Boundary-Aware U-Net with Edge-Sensitive Loss for Water Body Detection in Hyper-Arid Desert Regions

This repository contains the implementation of a deep learning framework for water body segmentation in hyper-arid desert environments using Boundary-Aware U-Net and comparative segmentation models.

## Features

- Progressive resizing training strategy
- Boundary-aware edge-sensitive loss
- Comparison of 10 segmentation architectures
- YOLOv8 object detection baseline
- Automatic checkpoint resume capability
- Full experiment result saving

## Models Included

- U-Net
- Boundary-Aware U-Net
- Attention U-Net
- U-Net++
- DeepLabV3+
- PSPNet
- FPN
- PAN
- LinkNet
- MAnet
- YOLOv8

## Requirements

Python 3.9+

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Structure

```text
data/
├── image/
└── mask/
```

## Training

Run:

```bash
python main.py
```

## Outputs

Results are automatically saved in:

```text
experiment_results/
```

Including:
- trained models
- segmentation predictions
- YOLO detections
- evaluation metrics
- plots and visualizations

## Example Test

Example sample images and masks are included in:

```text
sample_data/
```

## Citation

If you use this work, please cite:

Ashraf, H.A., "A Boundary-Aware U-Net with Edge-Sensitive Loss for Water Body Detection in Hyper-Arid Desert Region", Computers & Geosciences.

## License

This project is licensed under the MIT License.