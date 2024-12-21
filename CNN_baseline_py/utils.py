# utils.py

import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend suitable for script

import matplotlib.pyplot as plt
import os

import re
from datetime import datetime

def plot_loss(train_losses, val_losses, epoch, base_dir='results'):
    output_dir = os.path.join(base_dir, 'loss_plots')
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.title("Training and Validation Loss")
    plt.plot(range(1, epoch+2), train_losses, label="Train Loss")
    plt.plot(range(1, epoch+2), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plot_path = os.path.join(output_dir, f'loss_epoch_{epoch+1}.png')
    plt.savefig(plot_path)
    plt.close()

def plot_metrics(val_ious, val_dices, val_accuracies, epoch, base_dir='results'):
    output_dir = os.path.join(base_dir, 'metrics_plots')
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.title("Validation Metrics")
    epochs = range(1, epoch+2)
    plt.plot(epochs, val_ious, label="Validation IoU")
    plt.plot(epochs, val_dices, label="Validation Dice")
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Metric Value")
    plt.legend()
    plot_path = os.path.join(output_dir, f'metrics_epoch_{epoch+1}.png')
    plt.savefig(plot_path)
    plt.close()
    
def find_best_metrics(log_file):
    # Regular expression to extract metrics
    pattern = r"Epoch \[(\d+)/\d+\].*Val IoU: ([\d.]+), Val Dice: ([\d.]+), Val Accuracy: ([\d.]+)"
    
    max_sum = 0
    best_metrics = None
    
    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                epoch = int(match.group(1))
                iou = float(match.group(2))
                dice = float(match.group(3))
                acc = float(match.group(4))
                
                metric_sum = iou + dice + acc
                if metric_sum > max_sum:
                    max_sum = metric_sum
                    best_metrics = (epoch, iou, dice, acc, metric_sum)
    
    # Append result to log file
    if best_metrics:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        result = (f"{timestamp} Best metrics at epoch {best_metrics[0]}: "
                 f"Val IoU: {best_metrics[1]:.4f}, Val Dice: {best_metrics[2]:.4f}, "
                 f"Val Accuracy: {best_metrics[3]:.4f}\n")
        
        with open(log_file, 'a') as f:
            f.write(result)