from model import NeuralNetwork, Dense

class Trainer:
    """
    Trainer class, including: 
    - training loop
    - updating params
    """
    def __init__(self, model, learning_rate=0.01):
        self.model = model
        self.learning_rate = learning_rate

    def train(self, X, Y, num_iterations):
        for i in range(num_iterations):
            A = self.model.forward(X)
            loss = self.model.compute_loss(A, Y)
            if i == 0:
                print(f"Iteration 0, initial loss: {loss}")
            
            dA = self.model.compute_loss_gradient(A, Y)
            self.model.backward(dA)
            self.optim_step()

            if (i+1) % (num_iterations // 10) == 0:
                print(f"Iteration {i+1}, loss: {loss}")

    def optim_step(self):
        for layer in self.model.layers:
            if isinstance(layer, Dense):
                layer.update_params(self.learning_rate)
