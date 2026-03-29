import cupy as np


def one_hot(Y, num_classes):
    one_hot_Y = np.zeros((Y.size, num_classes))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, seedval=0):
        np.random.seed(seedval)
        self.weights = (
            np.random.randn(n_inputs, n_neurons).astype(np.float32)
            * np.sqrt(1.0 / n_inputs)
        ).astype(np.float32)
        self.biases = np.zeros((1, n_neurons), dtype=np.float32)

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        original_shape = self.inputs.shape
        inputs_2d  = self.inputs.reshape(-1, original_shape[-1])
        dvalues_2d = dvalues.reshape(-1, dvalues.shape[-1])

        self.dweights = inputs_2d.T @ dvalues_2d
        self.dbiases  = np.sum(dvalues_2d, axis=0, keepdims=True)

        dinputs_2d  = dvalues_2d @ self.weights.T
        self.dinputs = dinputs_2d.reshape(original_shape[:-1] + (self.weights.shape[0],))


class LayerNorm:
    def __init__(self, emb_dim):
        self.gamma = np.ones((1, 1, emb_dim),  dtype=np.float32)
        self.beta  = np.zeros((1, 1, emb_dim), dtype=np.float32)
        self.eps   = 1e-6

    def forward(self, x):
        self.x    = x
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var  = np.var(x,  axis=-1, keepdims=True)

        self.x_hat = (x - self.mean) / np.sqrt(self.var + self.eps)
        self.output = self.gamma * self.x_hat + self.beta
        return self.output

    def backward(self, dvalues):
        B, T, D = dvalues.shape
        N = D

        self.dgamma = np.sum(dvalues * self.x_hat, axis=(0, 1), keepdims=True)
        self.dbeta  = np.sum(dvalues,               axis=(0, 1), keepdims=True)

        dx_hat          = dvalues * self.gamma
        inv_std         = 1.0 / np.sqrt(self.var + self.eps)
        sum_dxhat       = np.sum(dx_hat,            axis=-1, keepdims=True)
        sum_dxhat_xhat  = np.sum(dx_hat * self.x_hat, axis=-1, keepdims=True)

        self.dinputs = (1.0 / N) * inv_std * (
            N * dx_hat - sum_dxhat - self.x_hat * sum_dxhat_xhat
        )
        return self.dinputs