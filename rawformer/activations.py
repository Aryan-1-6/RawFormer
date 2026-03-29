import cupy as np


class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Activation_Leaky_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0.2 * inputs, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] *= 0.2   # scale gradient, not replace


class Activation_Softmax:
    def forward(self, inputs):
        # Numerically stable softmax
        exp_values   = np.exp(inputs - np.max(inputs, axis=-1, keepdims=True))
        probabilities = exp_values / (np.sum(exp_values, axis=-1, keepdims=True) + 1e-10)
        self.output  = probabilities

    def backward(self, dvalues):
        s = self.output
        dot = np.sum(dvalues * s, axis=-1, keepdims=True)
        self.dinputs = s * (dvalues - dot)