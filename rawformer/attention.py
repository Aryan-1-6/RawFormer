import cupy as np
from rawformer.layers import Layer_Dense
from rawformer.activations import Activation_Softmax


class SelfAttention:
    def __init__(self, embd_dim, context):
        self.embd_dim  = embd_dim
        self.scale     = 1.0 / np.sqrt(embd_dim)

        # Fused QKV projection — 3x fewer kernel launches vs separate Q, K, V layers
        self.qkv_layer = Layer_Dense(embd_dim, 3 * embd_dim)
        self.softmax   = Activation_Softmax()

        # Causal mask: upper triangle = -1e9, lower = 0
        mask = np.triu(np.ones((context, context)), k=1)
        mask = np.where(mask == 1, -1e9, 0.0)
        self.mask = mask[np.newaxis, :, :]   # (1, T, T)

    def get_layers(self):
        return [self.qkv_layer]

    def forward(self, x):
        B, T, D = x.shape

        self.qkv_layer.forward(x)                                       # (B, T, 3*D)
        Q, K, V = np.split(self.qkv_layer.output, 3, axis=-1)          # each (B, T, D)
        self.Q, self.K, self.V = Q, K, V                                # save for backward

        scores = np.matmul(Q, K.transpose(0, 2, 1)) * self.scale       # (B, T, T)
        scores = scores + self.mask[:, :T, :T]                          # causal mask

        self.softmax.forward(scores)
        self.attn_weights = self.softmax.output                         # (B, T, T)

        attention = np.matmul(self.attn_weights, V)                     # (B, T, D)
        return attention

    def backward(self, dvalues):
        B, T, D = dvalues.shape

        # Gradient w.r.t V
        # attn_out = softmax_weights @ V  →  dV = softmax_weights.T @ d_attn_out
        dV = np.matmul(self.attn_weights.transpose(0, 2, 1), dvalues)  # (B, T, D)

        # Gradient w.r.t attention weights
        # d_attn_weights = d_attn_out @ V.T
        d_attn_weights = np.matmul(dvalues, self.V.transpose(0, 2, 1)) # (B, T, T)

        # Backprop through softmax
        self.softmax.backward(d_attn_weights)
        d_scores = self.softmax.dinputs                                  # (B, T, T)

        # Zero out gradients at masked positions
        d_scores *= (self.mask[:, :T, :T] > -1e8)

        # Gradients w.r.t Q and K
        # scores = Q @ K.T * scale  →  dQ = d_scores @ K * scale
        #                               dK = d_scores.T @ Q * scale
        dQ = np.matmul(d_scores, self.K) * self.scale                   # (B, T, D)
        dK = np.matmul(d_scores.transpose(0, 2, 1), self.Q) * self.scale # (B, T, D)

        # Concatenate dQ, dK, dV — reverse of np.split in forward
        d_qkv = np.concatenate([dQ, dK, dV], axis=-1)                  # (B, T, 3*D)

        # Backprop through fused QKV layer
        self.qkv_layer.backward(d_qkv)

        return self.qkv_layer.dinputs                                    # (B, T, D)