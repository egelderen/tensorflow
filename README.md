# Simple TensorFlow XOR Model

This is a basic example of a neural network built with TensorFlow to solve the XOR problem—a classic non-linear classification task. It demonstrates supervised learning with a small dataset using a multi-layer neural network.

## What It Does
The model takes pairs of binary inputs (e.g., `[0, 0]`, `[0, 1]`) and predicts their XOR output (`0` or `1`). It uses three layers to learn the pattern, showing how neural networks handle non-linear relationships.

- **Input**: Two binary numbers (e.g., `[1, 0]`).
- **Output**: A single binary value (e.g., `1`).
- **Example**: `[0, 0] → 0`, `[0, 1] → 1`, `[1, 0] → 1`, `[1, 1] → 0`.

## How It Works
- **Framework**: TensorFlow with Keras.
- **Architecture**: A sequential model with:
  - Input layer: 2 features.
  - Hidden layer 1: 8 neurons, ReLU activation.
  - Hidden layer 2: 4 neurons, ReLU activation.
  - Output layer: 1 neuron, sigmoid activation.
- **Training**: 1000 epochs, Adam optimizer (learning rate 0.01), binary crossentropy loss.
- **Data**: 4 hardcoded input-output pairs.

## Requirements
- Python 3.x
- TensorFlow (`pip install tensorflow`)
- NumPy (`pip install numpy`)
