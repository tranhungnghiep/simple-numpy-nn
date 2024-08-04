# Simple Neural Networks with NumPy
This is a neural networks implementation purely in NumPy, written for educational purpose with simple design choices and extensive comments. It demonstrates the fundamentals of neural networks, including forward-backward propagation and modular architecture design, without using high-level libraries like TensorFlow or PyTorch.

## Features
- Forward and Backward computation: Implements basic forward and backward computation with chain rule.
- Modular Layer-Based Architecture: Easily stacking Dense layers, ReLU, and Sigmoid activations.
- Loss Functions: Supports Mean Squared Error (MSE) and Cross-Entropy loss functions.
- Gradient Descent Optimization: Implements basic training loop with gradient descent to update parameters.
- Command-Line Interface: Easy to run and test different configurations through CLI.

## Getting Started
### Prerequisites
- Python 3
- NumPy

### Installation
Clone the repository to your local machine:
```
git clone https://github.com/tranhungnghiep/simple-numpy-nn.git
cd simple-numpy-nn
```

### Usage
Run the network training using the following command:
```
python main.py --n_features 3 --d_hidden 20 --n_examples 10 --num_iterations 100000 --learning_rate 0.01 --loss_function cross_entropy
```

The command line arguments include:
- --n_features: Number of input features.
- --d_hidden: Number of hidden units in the dense layer.
- --n_examples: Number of training examples.
- --num_iterations: Number of iterations to train the network.
- --learning_rate: Learning rate for training.
- --loss_function: Type of loss function to use (mse or cross_entropy).

## License
This project is open-sourced under the MIT license.

Contributions are welcome.
