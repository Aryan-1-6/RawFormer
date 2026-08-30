import cupy as np
from rawformer import Layer_Dense, Activation_ReLU
from time import perf_counter

class FeedForward:
    def __init__(self, embd_dim, **kwargs):
        # Standard 4x expansion as in "Attention Is All You Need"
        self.fc1 = Layer_Dense(embd_dim, embd_dim * 4)
        self.act = Activation_ReLU()
        self.fc2 = Layer_Dense(embd_dim * 4, embd_dim)
        self.start = 0
        self.debug = False
        if kwargs['DEBUG']:
            self.debug = kwargs['DEBUG']['ffn']

    def get_layers(self):
        return [self.fc1, self.fc2]

    def forward(self, x):
        if self.debug : self.start = perf_counter()

        self.fc1.forward(x)
        self.act.forward(self.fc1.output)
        self.fc2.forward(self.act.output)\

        if self.debug : print(f"FFN - Forward time : {perf_counter() - self.start}")

        return self.fc2.output

    def backward(self, dvalues):
        if self.debug : self.start = perf_counter()

        self.fc2.backward(dvalues)
        self.act.backward(self.fc2.dinputs)
        self.fc1.backward(self.act.dinputs)

        if self.debug : print(f"FFN : Backward time : {perf_counter() - self.start}")

        return self.fc1.dinputs