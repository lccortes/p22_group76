# dataset.py

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

class CellSegmentationDataset(Dataset):
    def __init__(self, dataframe, wells, transform=None, mask_transform=None, augmentation=False, focal_planes=None):
        """
        Args:
            dataframe (pd.DataFrame): The DataFrame containing image and mask information.
            wells (list): List of wells to include.
            transform (callable, optional): Optional transform to be applied to images.
            mask_transform (callable, optional): Optional transform to be applied to masks.
            augmentation (bool): Whether to apply data augmentation.
            image_level_transform (callable, optional): Transforms applied to individual images.
            focal_planes (list, optional): List of z_positions (focal planes) to include.
        """
        self.transform = transform
        self.mask_transform = mask_transform
        self.augmentation = augmentation
        self.focal_planes = focal_planes  # New parameter to control focal planes

        # Filter the DataFrame for the specified wells
        well_numbers = [num for num in wells]
        df_filtered = dataframe[dataframe['well_number'].isin(well_numbers)]

        # Get unique samples
        self.samples = self._get_samples(df_filtered)

    def _get_samples(self, df):
        samples = []
        # Group by unique samples (well_number, location_number, sample_site)
        grouped = df.groupby(['well_number', 'location_number', 'sample_site'])
        for _, group in grouped:
            # Get images (is_mask == False)
            images_df = group[group['is_mask'] == False]

            # Filter images based on focal planes if specified
            if self.focal_planes is not None:
                images_df = images_df[images_df['z_position'].isin(self.focal_planes)]
                # Ensure we have images for all specified focal planes
                if len(images_df) != len(self.focal_planes):
                    continue  # Skip samples that don't have images for all specified focal planes
                # Sort images according to the order of focal planes specified
                images_df['z_position'] = pd.Categorical(images_df['z_position'], categories=self.focal_planes, ordered=True)
                images_df = images_df.sort_values('z_position')
            else:
                # Ensure we have all 11 focal planes
                if len(images_df) != 11:
                    continue  # Skip samples that don't have all 11 images
                # Sort images by z_position
                images_df = images_df.sort_values('z_position')

            image_paths = images_df['path'].tolist()

            # Get the mask (is_mask == True)
            mask_df = group[group['is_mask'] == True]
            if len(mask_df) != 1:
                continue  # Skip if we don't have exactly one mask
            mask_path = mask_df['path'].values[0]

            samples.append((image_paths, mask_path))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_paths, mask_path = self.samples[idx]

        # Load images for the specified focal lengths as PIL Images
        images = []
        for img_path in img_paths:
            image = Image.open(img_path).convert('RGB')
            images.append(image)

        # Load mask as PIL Image
        mask = Image.open(mask_path).convert('L')

        # Apply data augmentation
        if self.augmentation:
            # Random Horizontal Flip
            if np.random.random() > 0.5:
                images = [F.hflip(img) for img in images]
                mask = F.hflip(mask)

            # Random Vertical Flip
            if np.random.random() > 0.5:
                images = [F.vflip(img) for img in images]
                mask = F.vflip(mask)

            # Random Rotation
            angle = np.random.uniform(-30, 30)
            images = [F.rotate(img, angle, interpolation=transforms.InterpolationMode.BILINEAR) for img in images]
            mask = F.rotate(mask, angle, interpolation=transforms.InterpolationMode.NEAREST)

        # Apply transforms to images
        if self.transform:
            images = [self.transform(img) for img in images]
        else:
            images = [transforms.ToTensor()(img) for img in images]

        # Concatenate images along the channel dimension
        image = torch.cat(images, dim=0)  # Shape: [11*C, H, W]

        # Apply transforms to mask
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask = transforms.ToTensor()(mask)

        # Ensure mask is binary
        mask = (mask > 0.5).float()

        return image, mask
