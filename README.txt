---

# Cell Segmentation using PyTorch

This project implements a baseline Convolutional Neural Network (CNN) for image segmentation of cell images taken at 11 different focal lengths. The goal is to segment the images to produce binary masks that highlight the locations of the cells.

## Table of Contents

- Overview
- Project Structure
- Dependencies
- Installation
- Data Preparation
- Usage
  - Training
  - Visualization
- Configuration
- Notes
- Acknowledgments

---

## Overview

The project is designed to:

- Train a CNN model using PyTorch for cell image segmentation.
- Handle input images taken at 11 different focal lengths, combining them into a single input tensor.
- Use a custom dataset class to efficiently load images and their corresponding masks based on a provided DataFrame.
- Include options to specify which wells to use for training and validation.
- Support data augmentation to improve model generalization.
- Display training and validation progress through loss graphs.
- Provide modular code for easy modification and expansion.

---

## Project Structure

```
├── dataset.py          # Contains the CellSegmentationDataset class
├── model.py            # Defines the UNet model
├── train.py            # Implements the training loop
├── utils.py            # Contains utility functions (e.g., plot_loss)
├── main.py             # Main script to run the training process
├── visualize.py        # (Optional) Contains the visualize_predictions function
├── README.txt          # Project documentation
└── requirements.txt    # List of required Python packages
```

---

## Dependencies

The project requires the following Python packages:

- Python 3.6 or higher
- torch
- torchvision
- pandas
- numpy
- matplotlib
- pillow
- tqdm

All dependencies are listed in the `requirements.txt` file.

---

## Installation

1. **Clone the Repository**

   ```
   git clone https://github.com/your_username/cell-segmentation-pytorch.git
   cd cell-segmentation-pytorch
   ```

2. **Create a Virtual Environment**

   It's recommended to use a virtual environment to manage dependencies.

   ```
   python -m venv venv
   source venv/bin/activate   # On Windows, use 'venv\Scripts\activate'
   ```

3. **Install Dependencies**

   ```
   pip install -r requirements.txt
   ```

---

## Data Preparation

1. **Data Directory Structure**

   The data directory should be structured as follows:

   ```
   data/
   └── brightfield/
       ├── Alexa488_Fibroblasts_well1_50locations/
       │   ├── images/
       │   └── masks/
       ├── Alexa488_Fibroblasts_well2_200locations/
       │   ├── images/
       │   └── masks/
       └── ... (other well directories)
   ```

2. **DataFrame CSV File**

   - Ensure you have a CSV file (e.g., `filename_data.csv`) that contains the image and mask information.
   - The CSV file should include the following columns:

     - `name`
     - `cell_type`
     - `well_number`
     - `location_number`
     - `tag`
     - `data_type`
     - `file_format`
     - `path`
     - `filename`
     - `subfolder`
     - `sample_site`
     - `z_position`
     - `channel`
     - `has_mask`

   - Place the CSV file in the `data/` directory or update the path accordingly in `main.py`.

3. **Verify Paths**

   - Ensure that the `path` column in the CSV file contains correct paths to the image files.
   - Update the `data_dir` and `dataframe_path` variables in `main.py` to match your data locations.

---

## Usage

### Training

1. **Configure Parameters**

   Open `main.py` and adjust the configuration parameters if necessary:

   ```python
   # main.py

   data_dir = "data/brightfield"            # Path to your data directory
   dataframe_path = "data/filename_data.csv"  # Path to your DataFrame CSV file
   wells_train = [2, 3, 4, 5, 6, 7]         # Wells to use for training
   wells_val = [1]                          # Wells to use for validation
   batch_size = 4
   epochs = 10
   augmentation = True                      # Toggle data augmentation
   learning_rate = 0.001
   ```

2. **Run Training**

   Execute the main script:

   ```
   python main.py
   ```

   The script will:

   - Load the data using the custom dataset.
   - Initialize the UNet model.
   - Train the model, displaying training and validation loss graphs after each epoch.

### Visualization

To visualize predictions after training:

1. **Modify `main.py`**

   Add the following lines at the end of `main.py`:

   ```python
   # main.py

   from visualize import visualize_predictions

   # After training
   visualize_predictions(trained_model, val_dataset, device, num_samples=5)
   ```

2. **Run the Script Again**

   ```
   python main.py
   ```

   This will display sample input images, their ground truth masks, and the predicted masks.

---

## Configuration

- **Data Augmentation**

  Data augmentation can be toggled using the `augmentation` parameter in `main.py`. Augmentations applied include:

  - Random horizontal flip
  - Random vertical flip
  - Random rotation (up to 30 degrees)

- **Hyperparameters**

  Adjust hyperparameters such as `batch_size`, `epochs`, and `learning_rate` in `main.py` to suit your needs.

- **Model Parameters**

  The UNet model is defined in `model.py` with default parameters. You can modify the architecture by changing the number of layers, filters, etc.

- **Transforms**

  Image and mask transformations are defined in `main.py`:

  ```python
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
  ])
  ```

---

## Notes

- **Data Consistency**

  The dataset class skips samples that don't meet the criteria (e.g., not having exactly 11 images or lacking a corresponding mask). Ensure your data is consistent to maximize the number of training samples.

- **Model Saving**

  The trained model can be saved by modifying `train.py`:

  ```python
  # At the end of train_model function in train.py
  torch.save(model.state_dict(), 'trained_model.pth')
  ```

- **GPU Acceleration**

  If you have a CUDA-compatible GPU, the script will automatically utilize it for training. Ensure that PyTorch is installed with GPU support.

- **Error Handling**

  Pay attention to any warnings or errors during data loading and training. Missing files or incorrect paths will lead to exceptions.

---

## Acknowledgments

This project was developed to assist in cell image segmentation tasks using PyTorch. The UNet architecture is widely used for image segmentation and is adapted here for the specific needs of handling multiple focal lengths.

---