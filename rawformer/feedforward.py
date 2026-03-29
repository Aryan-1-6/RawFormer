import cupy as np
from rawformer.layers import Layer_Dense
from rawformer.activations import Activation_ReLU


class FeedForward:
    def __init__(self, embd_dim):
        # Standard 4x expansion as in "Attention Is All You Need"
        self.fc1 = Layer_Dense(embd_dim, embd_dim * 4)
        self.act = Activation_ReLU()
        self.fc2 = Layer_Dense(embd_dim * 4, embd_dim)

    def get_layers(self):
        return [self.fc1, self.fc2]

    def forward(self, x):
        self.fc1.forward(x)
        self.act.forward(self.fc1.output)
        self.fc2.forward(self.act.output)
        return self.fc2.output

    def backward(self, dvalues):
        self.fc2.backward(dvalues)
        self.act.backward(self.fc2.dinputs)
        self.fc1.backward(self.act.dinputs)
        return self.fc1.dinputs