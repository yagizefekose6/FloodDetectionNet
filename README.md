# FloodDetectionNet 🌊

![Flood Detection](https://img.shields.io/badge/Flood%20Detection-U--Net%20with%20Attention%20Mechanism-brightgreen)

## Overview

FloodDetectionNet is a deep learning project focused on detecting floods using satellite imagery. This repository implements a U-Net architecture enhanced with an attention mechanism to improve segmentation accuracy. By leveraging state-of-the-art techniques in computer vision, we aim to contribute to disaster management and response efforts.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

- **Attention Mechanism**: Enhances the U-Net architecture by focusing on relevant features.
- **Data Augmentation**: Improves model robustness through various augmentation techniques.
- **Deep Learning Framework**: Built on TensorFlow for efficient training and deployment.
- **Image Segmentation**: Provides pixel-level classification for accurate flood detection.
- **Disaster Management**: Supports timely and effective responses to flood events.

## Installation

To set up the FloodDetectionNet project, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/yagizefekose6/FloodDetectionNet.git
   cd FloodDetectionNet
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

After installation, you can run the model with the following command:

```bash
python main.py --input <path_to_input_image> --output <path_to_output_image>
```

Replace `<path_to_input_image>` with the path to your satellite image and `<path_to_output_image>` with the desired output path for the segmented image.

For pre-trained models and releases, visit the [Releases section](https://github.com/yagizefekose6/FloodDetectionNet/releases). Download the necessary files and execute them as needed.

## Dataset

FloodDetectionNet utilizes satellite imagery datasets for training and evaluation. We recommend using publicly available datasets such as:

- [NASA's MODIS](https://modis.gsfc.nasa.gov/data/dataprod/mod09/)
- [Sentinel-2](https://scihub.copernicus.eu/dhus)

Ensure to preprocess the data as required by the model.

## Model Architecture

FloodDetectionNet employs a U-Net architecture enhanced with an attention mechanism. The architecture consists of:

- **Encoder**: Down-sampling layers that capture context.
- **Bottleneck**: The deepest layer that captures the most abstract features.
- **Decoder**: Up-sampling layers that reconstruct the image.
- **Attention Gates**: Focus on important features, improving segmentation quality.

![U-Net Architecture](https://miro.medium.com/v2/resize:fit:1200/format:webp/1*I6O0zC9VwB0V3f8qVYxkgw.png)

## Training

To train the model, use the following command:

```bash
python train.py --epochs <number_of_epochs> --batch_size <batch_size>
```

Adjust `<number_of_epochs>` and `<batch_size>` as needed. Monitor the training process through logs generated in the console.

## Evaluation

After training, evaluate the model using:

```bash
python evaluate.py --model <path_to_trained_model> --test_data <path_to_test_data>
```

This will provide metrics such as accuracy, precision, recall, and F1-score.

## Results

The model achieves promising results in detecting floods. Here are some example outputs:

![Flood Detection Example](https://example.com/flood_detection_example.png)

You can find more results in the `results` folder.

## Contributing

We welcome contributions! If you have suggestions or improvements, please fork the repository and submit a pull request. Make sure to follow the coding standards and include tests for new features.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For questions or feedback, feel free to reach out:

- **Author**: [Your Name](https://github.com/yourusername)
- **Email**: your.email@example.com

For additional resources, check the [Releases section](https://github.com/yagizefekose6/FloodDetectionNet/releases) for model files and updates.