# Week 4 Solution - Deep Learning with PyTorch

## Introduction
This project is part of the **Arewa Data Science Academy's Deep Learning Cohort 1.0** program. It focuses on implementing a convolutional neural network (CNN) using PyTorch to classify handwritten digits from the MNIST dataset. The tasks include loading and visualizing the dataset, building a CNN model, training it on both CPU and GPU, and evaluating its performance using metrics such as accuracy and confusion matrices.

## Process
The following steps were followed to complete the exercises:
1. **Dataset Preparation**:
   - The MNIST dataset was loaded using `torchvision.datasets.MNIST`.
   - Training and testing datasets were transformed into tensors using `ToTensor()` and visualized.

2. **Data Loading**:
   - The datasets were converted into dataloaders using `torch.utils.data.DataLoader` with a batch size of 32.

3. **Model Creation**:
   - A CNN model, inspired by TinyVGG, was implemented using PyTorch's `nn.Module`.
   - The model architecture included convolutional layers, ReLU activations, max pooling, and a fully connected layer.

4. **Training**:
   - The model was trained on both CPU and GPU for 3 epochs.
   - Training and testing steps were implemented to calculate loss and accuracy.

5. **Evaluation**:
   - Predictions were made on test samples, and the results were visualized.
   - A confusion matrix was plotted to compare the model's predictions with the true labels.

6. **Exploration**:
   - Additional experiments were conducted, such as varying kernel sizes in convolutional layers and analyzing model errors.

## Tech Tools
The following tools and technologies were used in this project:
- **Python**: The programming language used for all implementations.
- **PyTorch**: The deep learning framework used for building and training the CNN model.
- **Torchvision**: Used for loading and transforming the MNIST dataset.
- **Matplotlib**: A library used for visualizing the data and predictions.
- **TorchMetrics**: Used for calculating metrics such as confusion matrices.
- **Jupyter Notebook**: An interactive environment for writing and running Python code.

## Conclusion
This project demonstrates the process of building, training, and evaluating a convolutional neural network using PyTorch. By completing these exercises, a solid understanding of CNNs and their application to image classification tasks was achieved. The outcomes include a trained model capable of classifying handwritten digits with high accuracy and insights into model performance through visualizations and metrics.

### Relevant Tags
- PyTorch
- Convolutional Neural Networks
- Image Classification
- MNIST Dataset
- Deep Learning
- Machine Learning
- Arewa Data Science Academy
- Python Programming