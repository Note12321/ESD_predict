# ESD Video Analysis Project

This project aims to analyze ESD (Electrostatic Discharge) surgical videos using the YOLOv8 model for video classification into 8 distinct categories. The project includes various components such as configuration files, data loaders, model definitions, and training scripts.

## Project Structure

```
ESD_video_analysis
├── config
│   └── config.yaml
├── data_loader
│   └── data_loader.py
├── module
│   └── module.py
├── train
│   └── train.py
├── utils
│   └── utils.py
├── models
│   └── yolo_v8.py
├── requirements.txt
└── README.md
```

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```
   git clone <repository_url>
   cd ESD_video_analysis
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Configuration

The configuration for the YOLOv8 model can be found in `config/config.yaml`. This file includes hyperparameters, category definitions, and paths for training and validation datasets.

## Data Loading

The data loader is implemented in `data_loader/data_loader.py`. It is responsible for reading video files and their corresponding annotation files, performing data preprocessing and augmentation, and preparing the dataset for training.

## Model Definition

The YOLOv8 model is encapsulated in `module/module.py`, which defines the model architecture, forward propagation methods, and inference functionalities.

## Training

To train the model, run the training script located in `train/train.py`. This script will load the configuration, initialize the data loader and model, set up the training loop, and save the trained model.

## Utilities

Utility functions, such as calculating mean Average Precision (mAP) and plotting loss curves during training, are provided in `utils/utils.py`.

## YOLOv8 Model Implementation

The specific structure of the YOLOv8 model, including convolutional layers, activation functions, and loss functions, can be found in `models/yolo_v8.py`.

## Usage

After training the model, you can use it for inference on new video data. Refer to the documentation in the respective files for detailed usage instructions.

## License

This project is licensed under the MIT License - see the LICENSE file for details.