import numpy as np


class ReLU:
    """
    ReLU activation function as a layer.
    """
    def forward(self, Z):
        self.Z = Z
        return np.maximum(0, Z)

    def backward(self, dA):
        dZ = dA * (self.Z > 0)
        return dZ


class Sigmoid:
    """
    Sigmoid activation function as a layer.
    """
    def forward(self, Z):
        self.Z = Z

        # Overflow when Z < 0, because np.exp(a large positive number):
        # self.A = 1 / (1 + np.exp(-Z))

        # Numerical stable sigmoid computation, using alternative form when Z < 0:
        self.A = np.where(
            Z >= 0, # condition
            1 / (1 + np.exp(-Z)), # For positive Z
            np.exp(Z) / (1 + np.exp(Z)) # For negative Z
        )
        
        return self.A
    
    def backward(self, dA):
        dSigmoid = self.A * (1 - self.A)  # d s(x) = s(x) * (1 - s(x))
        dZ = dA * dSigmoid
        return dZ


class Dense:
    """
    Fully connected layer.
    Notation: 
    - Z: pre-activation output value
    - A: post-activation output value
    - dA: gradient wrt post-activation output
    - dZ: gradient wrt pre-activation output
    """
    def __init__(self, n_input, n_output):
        self.W = np.random.randn(n_input, n_output) * 0.01  # (i, o)
        self.b = np.zeros((1, n_output))  # (1, o)

    def forward(self, A):
        """
        A: (b, i)
        A * W -> Z
        """
        self.A_prev = A
        self.Z = np.dot(A, self.W) + self.b  # (b, i) * (i, o) + (1, o) -> (b, o)
        return self.Z

    def backward(self, dZ):
        """
        dZ: (b, o)
        TODO: 
        - compute gradient for L2 weight decay: dW += (lambda_l2 / b) * W
        """
        # Compute gradients to update parameters
        # A.T * dZ -> dW (need cached input A)
        b = self.A_prev.shape[0]  # batch size b
        self.dW = (1 / b) * np.dot(self.A_prev.T, dZ)  # (i, b) * (b, o) -> (i, o)
        self.db = (1 / b) * np.sum(dZ, axis=0, keepdims=True)  # (b, o) -> (1, o)

        # Backward gradient using current parameter values before updating
        # dZ * W.T -> dA_previous (backward dA for previous layer)
        self.dA_prev = np.dot(dZ, self.W.T)  # (b, o) * (o, i) -> (b, i)

        return self.dA_prev
    
    def update_params(self, learning_rate):
        """
        Update params using cached gradients dW, db
        """
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db


class NeuralNetwork:
    """
    Neural network class, including: 
    - all layers forward
    - all layers backward
    - loss function
    - loss gradient
    """
    def __init__(self, layers, loss_function='mse'):
        self.layers = layers
        self.loss_function = loss_function

    def forward(self, X):
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def backward(self, dLoss):
        dA = dLoss
        for layer in reversed(self.layers):
            dA = layer.backward(dA)

    def compute_loss(self, A, Y, epsilon = 1e-15):
        b = Y.shape[0]
        if self.loss_function == 'mse':
            loss = (1 / (2 * b)) * np.sum((A - Y) ** 2)
        elif self.loss_function == 'cross_entropy':
            A = np.clip(A, epsilon, 1 - epsilon)  # prevent log(0)
            loss = -(1 / b) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
        return loss
    
    def compute_loss_gradient(self, A, Y, epsilon = 1e-15):
        if self.loss_function == 'mse':
            dA = A - Y
        elif self.loss_function == 'cross_entropy':
            A = np.clip(A, epsilon, 1 - epsilon)  # prevent division by 0
            dA = (A - Y) / (A * (1 - A))
            # Note: the exact dL/dA gradient has the division term (A*(1-A)).
            # However, the same term appear in the sigmoid gradient dA/dZ.
            # Thus, they cancel: dL/dZ = dL/dA * dA/dZ 
            #   = (A - Y) / (A*(1-A)) * (A*(1-A)) = (A - Y).
            # In practice, we could merge and omit both (A*(1-A)) terms
            #   and compute directly dL/dZ = (A - Y).
        return dA
