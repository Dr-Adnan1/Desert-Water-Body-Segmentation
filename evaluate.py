import numpy as np
from scipy import ndimage


def post_process(mask, min_area=100):
    labeled, num = ndimage.label(mask)

    sizes = ndimage.sum(mask, labeled, range(1, num + 1))

    result = np.zeros_like(mask)

    for i, size in enumerate(sizes, 1):
        if size >= min_area:
            result[labeled == i] = 1

    return result


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

    kernel = np.ones((5, 5), dtype=np.uint8)

    pred_boundary = ndimage.binary_dilation(pred_mask, kernel) ^ ndimage.binary_erosion(pred_mask, kernel)
    gt_boundary = ndimage.binary_dilation(gt_mask, kernel) ^ ndimage.binary_erosion(gt_mask, kernel)

    boundary_iou = (
        np.sum(pred_boundary & gt_boundary) + 1e-6
    ) / (
        np.sum(pred_boundary | gt_boundary) + 1e-6
    )

    return dice, iou, precision, recall, f1, boundary_iou