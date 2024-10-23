"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

import minitorch
import random

def RParam(*shape):
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)

class Network(minitorch.Module):
    def __init__(self, hidden_layers):
        super().__init__()
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x):
        middle = self.layer1.forward(x).relu()
        hidden = self.layer2.forward(middle).relu()
        output = self.layer3.forward(hidden).sigmoid()
        return output

class Linear (minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        backend = minitorch.TensorBackend(minitorch.SimpleOps)
        random_weights = [(2 * (random.random() - 0.5)) for _ in range(in_size * out_size)]
        random_bias = [2 * (random.random() - 0.5) for _ in range(out_size)]
        self.weights = self.add_parameter("weight", minitorch.Tensor.make(random_weights, (in_size, out_size), backend=backend))
        self.bias = self.add_parameter("bias", minitorch.Tensor.make(random_bias, (1, out_size), backend=backend))

    def forward(self, x: minitorch.Tensor):
        x_reshaped = x.view(*x.shape, 1)
        product = x_reshaped * self.weights.value
        result = product.sum(1).view(product.shape[0], product.shape[2])
        result = result + self.bias.value
        return result

def default_log_fn(epoch, total_loss, correct, losses):
    print(f"Epoch {epoch}, Loss: {total_loss:.4f}, Correct: {correct}")

class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        print(f"Training data X shape: {X.shape}")
        print(f"Training data y shape: {y.shape}")

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            if epoch % 50 == 0:  # Print every 50 epochs
                print(f"\nEpoch {epoch}:")
                print(f"Layer1 weight mean: {self.model.layer1.weights.value.sum()[0]}")
                print(f"Layer2 weight mean: {self.model.layer2.weights.value.sum()[0]}")
                print(f"Layer3 weight mean: {self.model.layer3.weights.value.sum()[0]}")

            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            if epoch % 50 == 0:
                print(f"Output mean: {out.sum()[0] / data.N}")

            prob = (out * y) + (out - 1.0) * (y - 1.0)
            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 8
    RATE = 0.1
    data = minitorch.datasets["Simple"](PTS)
    TensorTrain(HIDDEN).train(data, RATE)
