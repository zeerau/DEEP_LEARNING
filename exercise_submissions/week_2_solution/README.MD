# Week 2 Solution - Deep Learning with PyTorch

## Introduction
This project is part of the **Arewa Data Science Academy's Deep Learning Cohort 1.0** program. It focuses on implementing a linear regression model using PyTorch. The tasks include creating a dataset, building a custom PyTorch model, training the model, and saving/loading the model for inference. The goal is to gain hands-on experience with PyTorch's core functionalities and understand the workflow of building and training machine learning models.

## Process
The following steps were followed to complete the exercises:

1. **Dataset Creation**: 
   - A synthetic dataset was created using the linear regression formula `y = weight * X + bias` with `weight=0.3` and `bias=0.9`.
   - The dataset was split into 80% training and 20% testing data.

2. **Data Visualization**:
   - A helper function was implemented to visualize the training and testing data, along with predictions.

3. **Model Building**:
   - A custom linear regression model was built by subclassing `nn.Module`.
   - The model included `nn.Parameter` for weights and bias, and the `forward()` method implemented the linear regression formula.

4. **Training**:
   - The model was trained using the Mean Absolute Error (MAE) loss function (`nn.L1Loss`) and the Stochastic Gradient Descent (SGD) optimizer.
   - A training loop was implemented to train the model for 300 epochs, with evaluation on the test dataset every 20 epochs.

5. **Model Evaluation**:
   - The trained model's predictions were visualized against the original data to assess its performance.

6. **Model Saving and Loading**:
   - The model's `state_dict()` was saved to a file.
   - A new instance of the model was created, and the saved `state_dict()` was loaded into it.
   - Predictions from the loaded model were compared with the original model to ensure consistency.

## Tech Tools
The following tools and technologies were used in this project:
- **Python**: The programming language used for all implementations.
- **PyTorch**: The deep learning framework used for building and training the model.
- **Matplotlib**: A library used for visualizing the data and predictions.
- **Jupyter Notebook**: An interactive environment for writing and running Python code.

## Conclusion
This project demonstrates the end-to-end process of building, training, and evaluating a linear regression model using PyTorch. By completing these exercises, a solid understanding of PyTorch's basic functionalities was achieved, including dataset creation, model building, training loops, and model persistence. These skills serve as a foundation for more advanced deep learning tasks.

### Relevant Tags
- PyTorch
- Linear Regression
- Machine Learning
- Deep Learning
- Model Training
- Model Evaluation
- Arewa Data Science Academy
- Python Programming