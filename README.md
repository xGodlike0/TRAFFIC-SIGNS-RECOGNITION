# PyTorch CNN for Traffic Sign Classification

## Overview
This project presents a PyTorch implementation of a Convolutional Neural Network (CNN) for the task of traffic sign classification. The model is trained on a custom dataset of traffic sign images, using PyTorch for both model construction and training.

## Features
- **Data Handling**: Includes functions to load and split data into training, validation, and testing sets.
- **Custom PyTorch Dataset**: Implementation of a custom dataset class for handling image data within PyTorch.
- **Data Augmentation**: Utilizes data augmentation techniques for better model generalization.
- **CNN Architecture**: A CNN model with multiple convolutional and fully connected layers.
- **Training and Validation**: Functions for training the model and evaluating its performance on a validation dataset.

## Installation
To run this project, install the required libraries as listed below:
- PyTorch
- NumPy
- OpenCV
- scikit-learn

You can install these packages using `pip`:
```bash
pip install torch numpy opencv-python scikit-learn
```

## Usage
To use this project:
1. Clone the repository.
2. Place your dataset in the appropriate directory.
3. Run the Jupyter Notebook to train the model.

## Model Training
The training process involves several steps including loading data, applying transformations, and training the model over multiple epochs. The model's performance is evaluated at the end of each epoch.

## Real-time Testing
The script includes a section for real-time testing of the trained model using a webcam feed.

## Contributing
Feel free to fork this project and submit pull requests for any improvements or bug fixes.

## License
This project is open-source and available under the [MIT License](LICENSE.md).
