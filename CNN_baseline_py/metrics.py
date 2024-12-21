# metrics.py

import torch

def compute_iou(pred, target, threshold=0.5):
    """
    Computes the Intersection over Union (IoU) metric.
    """
    # Apply sigmoid activation to logits
    pred = torch.sigmoid(pred)
    pred = (pred >= threshold).float()
    target = target.float()
    # Flatten tensors
    pred = pred.view(-1)
    target = target.view(-1)
    # Intersection and Union
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection / union).item() if union > 0 else 1.0
    return iou

def compute_dice_coefficient(pred, target, threshold=0.5):
    """
    Computes the Dice coefficient.
    """
    # Apply sigmoid activation to logits
    pred = torch.sigmoid(pred)
    pred = (pred >= threshold).float()
    target = target.float()
    # Flatten tensors
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    dice = (2 * intersection) / (pred.sum() + target.sum()) if (pred.sum() + target.sum()) > 0 else 1.0
    return dice.item()

def compute_accuracy(pred, target, threshold=0.5):
    """
    Computes the Rand Index between predicted and target segmentation masks.
    Rand Index measures similarity between two clusterings, ranging from 0 to 1.
    """
    # Apply sigmoid activation to logits
    pred = torch.sigmoid(pred)
    pred = (pred >= threshold).float()
    target = target.float()
    # Flatten tensors
    pred = pred.view(-1)
    target = target.view(-1)
    # Calculate True Positives, True Negatives, False Positives, False Negatives
    tp = (pred * target).sum()
    tn = ((1 - pred) * (1 - target)).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    # Compute Rand Index
    rand_index = (tp + tn) / (tp + tn + fp + fn)
    return rand_index.item()
