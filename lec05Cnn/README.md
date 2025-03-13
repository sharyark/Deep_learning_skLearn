# CNN for MNIST using Keras

## Overview
This project implements a simple Convolutional Neural Network (CNN) using Keras to classify handwritten digits from the MNIST dataset. It includes stride and pooling layers to improve feature extraction and reduce spatial dimensions.

## Dependencies
Ensure you have the following dependencies installed:

```bash
pip install tensorflow matplotlib
```

## Model Architecture
1. **Conv2D Layer (32 filters, 3x3 kernel, stride=1, ReLU activation)**
2. **MaxPooling2D (2x2 pool size)**
3. **Conv2D Layer (64 filters, 3x3 kernel, stride=1, ReLU activation)**
4. **MaxPooling2D (2x2 pool size)**
5. **Flatten Layer**
6. **Dense Layer (64 neurons, ReLU activation)**
7. **Dense Output Layer (10 neurons, Softmax activation)**

## Training
Run the following command to train the model:

```python
python cnn_mnist_keras.py
```

The model is trained for 5 epochs with a batch size of 64.

## Evaluation
After training, the model is evaluated on the test dataset, and accuracy is printed. A plot of training and validation accuracy is also displayed.

## Output
- Training accuracy and validation accuracy
- Test accuracy
- Accuracy plot per epoch

