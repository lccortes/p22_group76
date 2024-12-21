# main.py

import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import time

from dataset import CellSegmentationDataset
from model import UNet
from train import train_model
from visualize import visualize_predictions

def main():
    # Configuration
    data_dir = "data/brightfield"  # Update this to your data directory path
    dataframe_path = "data/filename_data.csv"  # Path to your DataFrame CSV file
    wells_train = [2, 3, 4, 5, 6, 7]  # Wells to use for training
    wells_val = [1]  # Wells to use for validation
    batch_size = 16
    epochs = 50
    augmentation = True  # Toggle data augmentation
    learning_rate = 1e-4
    pos_weight = 1.0  # Adjusts how the model weighs positive samples (foreground)
    focal_planes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # Focal planes to use for training
    
    # Create a dictionary with the hyperparameters
    hyperparameters = {
        "wells_train": wells_train,
        "wells_val": wells_val,
        "batch_size": batch_size,
        "epochs": epochs,
        "augmentation": augmentation,
        "learning_rate": learning_rate,
        "pos_weight": pos_weight,
        "focal_planes": focal_planes
    }
    
    # Create base results directory at startup
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_dir = 'results' + os.sep + timestamp
    os.makedirs(base_dir, exist_ok=True)
    
    # Create model checkpoints directory
    checkpoint_dir = os.path.join(base_dir, 'model_checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save hyperparameters to a text file
    hyperparameters_txt = os.path.join(base_dir, 'hyperparameters.txt')
    with open(hyperparameters_txt, 'w') as f:
        for key, value in hyperparameters.items():
            f.write(f"{key}: {value}\n")

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the DataFrame
    df = pd.read_csv(dataframe_path, delimiter=';', index_col=0)

    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # Add other transformations if needed
    ])

    # Mask transformations
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # You might want to add transforms for masks if necessary
    ])

    # Datasets and Dataloaders
    train_dataset = CellSegmentationDataset(
        dataframe=df,
        wells=wells_train,
        transform=transform,
        mask_transform=mask_transform,
        augmentation=augmentation,
        focal_planes=focal_planes,
    )
    val_dataset = CellSegmentationDataset(
        dataframe=df,
        wells=wells_val,
        transform=transform,
        mask_transform=mask_transform,
        augmentation=False,
        focal_planes=focal_planes,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Optional: Test the dataset
    sample_image, sample_mask = train_dataset[0]
    print(f"Sample image shape: {sample_image.shape}")  # Should be [N*3, 256, 256]
    print(f"Sample mask shape: {sample_mask.shape}")    # Should be [1, 256, 256] 

    # Initialize Model
    in_channels = len(focal_planes) * 3  # 3 RGB channels per focal length
    model = UNet(in_channels, out_channels=1, dropout_prob=0.6).to(device)

    # Save model into a text file
    model_txt = os.path.join(base_dir, 'model.txt')
    with open(model_txt, 'w') as f:
        f.write(str(model))
        
    # Start Training
    trained_model = train_model(model, train_loader, val_loader, epochs, device, pos_weight, learning_rate, base_dir, optimizer='adam')
    
    # Visualize predictions
    visualize_predictions(trained_model, val_dataset, device, num_samples=5, base_dir=base_dir)

if __name__ == "__main__":
    main()
