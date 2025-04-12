# Week 3 Solution - Deep Learning with PyTorch

## Introduction
This project is part of the **Arewa Data Science Academy's Deep Learning Cohort 1.0** program. It focuses on implementing binary and multi-class classification models using PyTorch. The tasks include creating datasets, building custom PyTorch models, training the models, and visualizing decision boundaries. The goal is to gain hands-on experience with PyTorch's functionalities and understand the workflow of building and training machine learning models.

## Process
The following steps were followed to complete the exercises:

1. **Binary Classification**:
   - A synthetic dataset was created using Scikit-Learn's `make_moons()` function with 1000 samples.
   - The dataset was split into training and testing sets (80% training, 20% testing).
   - A binary classification model was built by subclassing `nn.Module` and incorporating non-linear activation functions.
   - The model was trained using the Binary Cross-Entropy with Logits Loss (`nn.BCEWithLogitsLoss`) and the Adam optimizer.
   - A training loop was implemented to train the model for 1000 epochs, achieving over 96% accuracy.
   - The decision boundaries of the trained model were visualized for both training and testing datasets.

2. **Multi-Class Classification**:
   - A spiral dataset was created using a custom implementation inspired by the CS231n course.
   - The dataset was split into training and testing sets (80% training, 20% testing).
   - A multi-class classification model was built using a combination of linear and non-linear layers with ReLU activation functions.
   - The model was trained using the Cross-Entropy Loss (`nn.CrossEntropyLoss`) and the Adam optimizer.
   - A training loop was implemented to train the model for 1000 epochs, achieving over 95% testing accuracy.
   - The decision boundaries of the trained model were visualized for both training and testing datasets.

3. **Custom Activation Function**:
   - The Tanh activation function was replicated in pure PyTorch to understand its mathematical implementation.

## Tech Tools
The following tools and technologies were used in this project:
- **Python**: The programming language used for all implementations.
- **PyTorch**: The deep learning framework used for building and training the models.
- **Scikit-Learn**: Used for generating synthetic datasets.
- **Matplotlib**: A library used for visualizing the data and decision boundaries.
- **TorchMetrics**: Used for calculating accuracy metrics.
- **Jupyter Notebook**: An interactive environment for writing and running Python code.

## Conclusion
This project demonstrates the end-to-end process of building, training, and evaluating binary and multi-class classification models using PyTorch. By completing these exercises, a solid understanding of PyTorch's functionalities was achieved, including dataset creation, model building, training loops, and visualization. These skills serve as a foundation for more advanced deep learning tasks.

### Relevant Tags
- PyTorch
- Binary Classification
- Multi-Class Classification
- Machine Learning
- Deep Learning
- Model Training
- Model Evaluation
- Arewa Data Science Academy
- Python Programming