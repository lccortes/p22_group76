# train.py

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim import SGD
from tqdm import tqdm
from utils import plot_loss, plot_metrics, find_best_metrics
from metrics import compute_iou, compute_dice_coefficient, compute_accuracy 
import logging
import os

# Define Dice Loss
def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)  # Apply sigmoid to logits
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice

def train_model(model, train_loader, val_loader, epochs, device, pos_weight, learning_rate, base_dir, optimizer='adam'):
    pos_weight = torch.tensor([pos_weight]).to(device)  # Adjust the weight as needed
    criterion_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = SGD(model.parameters(), lr=learning_rate)
    train_losses = []
    val_losses = []
    val_ious = []
    val_dices = []
    val_accuracies = []  # List to store accuracy values

    # Update paths
    log_path = os.path.join(base_dir, 'training.log')
    checkpoint_dir = os.path.join(base_dir, 'model_checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Set up logging
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s %(message)s')
    logging.info("Logging is configured correctly.")
    logging.info(f"Using device: {device}")
        
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion_bce(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        total_iou = 0.0
        total_dice = 0.0
        total_accuracy = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{epochs}"):
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion_bce(outputs, masks)
                val_loss += loss.item()

                # Compute metrics
                iou = compute_iou(outputs, masks)
                dice = compute_dice_coefficient(outputs, masks)
                accuracy = compute_accuracy(outputs, masks)
                total_iou += iou
                total_dice += dice
                total_accuracy += accuracy

        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = total_iou / len(val_loader)
        avg_val_dice = total_dice / len(val_loader)
        avg_val_accuracy = total_accuracy / len(val_loader)
        val_losses.append(avg_val_loss)
        val_ious.append(avg_val_iou)
        val_dices.append(avg_val_dice)
        val_accuracies.append(avg_val_accuracy)

        # Log the results
        logging.info(f"Epoch [{epoch+1}/{epochs}], "
                     f"Train Loss: {avg_train_loss:.4f}, "
                     f"Val Loss: {avg_val_loss:.4f}, "
                     f"Val IoU: {avg_val_iou:.4f}, "
                     f"Val Dice: {avg_val_dice:.4f}, "
                     f"Val Accuracy: {avg_val_accuracy:.4f}")

        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val IoU: {avg_val_iou:.4f}, "
              f"Val Dice: {avg_val_dice:.4f}, "
              f"Val Accuracy: {avg_val_accuracy:.4f}")

        # Plot losses and metrics after each epoch, passing the epoch number
        plot_loss(train_losses, val_losses, epoch, base_dir)
        plot_metrics(val_ious, val_dices, val_accuracies, epoch, base_dir)
        
        # Save model checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'trained_model_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        
    # Identify best metrics
    find_best_metrics(log_path) # Best metrics are printed in the final line of the log file

    return model