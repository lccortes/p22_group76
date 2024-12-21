# visualize.py

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend if running remotely

import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from metrics import compute_iou, compute_dice_coefficient, compute_accuracy  # Import metrics

def visualize_predictions(model, dataset, device, num_samples=5, base_dir='results'):
    output_dir = os.path.join(base_dir, 'predictions')
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    for i in range(num_samples):
        image, mask = dataset[i]
        image = image.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image)
        output = output.squeeze().cpu()
        mask = mask.squeeze()
        image_np = image.squeeze().cpu().numpy()

        # Apply threshold to obtain binary mask
        binary_output = (output >= 0.5).float()

        # Compute metrics for this sample
        iou = compute_iou(output, mask)
        dice = compute_dice_coefficient(output, mask)
        accuracy = compute_accuracy(output, mask)

        # Since image has multiple channels, select the first RGB image for visualization
        first_rgb = image_np[:3]  # First 3 channels correspond to the first focal length's RGB image

        # Plot the results
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.title('Input Image')
        plt.imshow(np.transpose(first_rgb, (1, 2, 0)))
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.title('Ground Truth')
        plt.imshow(mask.cpu().numpy(), cmap='gray')
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.title(f'Predicted Mask\nIoU: {iou:.4f}, Dice: {dice:.4f}, Acc: {accuracy:.4f}')
        plt.imshow(binary_output.cpu().numpy(), cmap='gray')
        plt.axis('off')

        # Save the figure
        plt.savefig(os.path.join(output_dir, f'prediction_{i+1}.png'))
        plt.close()