import numpy as np
from model import ReLU, Sigmoid, Dense, NeuralNetwork
from trainer import Trainer
import argparse


def main(args):
    np.random.seed(1)  # CAUTION: random seed affects convergence rate for small dataset
    X = np.random.randn(args.n_examples, args.n_features)  # (examples, features)
    if args.loss_function == 'mse':
        Y = np.random.randn(args.n_examples, 1)  # (examples, output): real output to test mse loss
    else:
        Y = np.random.randint(0, 2, (args.n_examples, 1))  # low 0, high 2-1, (examples, output): binary output to test cross-entropy loss

    layers = [
        Dense(args.n_features, args.d_hidden),
        ReLU(),
        Dense(args.d_hidden, 1),
    ]
    if args.loss_function == 'cross_entropy':
        layers.append(Sigmoid())  # binary classification with sigmoid output

    nn = NeuralNetwork(layers, loss_function=args.loss_function)
    trainer = Trainer(nn, learning_rate=args.learning_rate)
    print(f"Start training with {args.loss_function} loss.")
    trainer.train(X, Y, num_iterations=args.num_iterations)
    print(f"Training with {args.loss_function} loss complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a simple NN using numpy")
    parser.add_argument('--n_features', type=int, default=3, help='Number of input features')
    parser.add_argument('--d_hidden', type=int, default=20, help='Number of hidden units')
    parser.add_argument('--n_examples', type=int, default=10, help='Number of training examples')
    parser.add_argument('--num_iterations', type=int, default=100000, help='Number of iterations to train the network')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for training')
    parser.add_argument('--loss_function', type=str, choices=['mse', 'cross_entropy'], default='cross_entropy', help='Type of loss function to use')
    
    args = parser.parse_args()
    main(args)
