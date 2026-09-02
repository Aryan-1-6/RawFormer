import cupy as np
from rawformer import Layer_Dense, Activation_Softmax
from time import perf_counter

class SelfAttention:
    def __init__(self, embd_dim, context, n_heads, **kwargs):
        self.embd_dim  = embd_dim
        self.n_heads = n_heads
        self.scale     = 1.0 / np.sqrt(embd_dim)

        # Fused QKV projection — 3x fewer kernel launches vs separate Q, K, V layers
        self.qkv_layer = Layer_Dense(embd_dim, 3 * embd_dim)
        self.softmax   = Activation_Softmax()

        # Causal mask: upper triangle = -1e9, lower = 0
        mask = np.triu(np.ones((context, context)), k=1)
        mask = np.where(mask == 1, -1e9, 0.0)
        self.mask = mask[np.newaxis, np.newaxis, :, :]   # (1, 1, T, T)
        self.start = 0

        self.debug = False
        if kwargs['DEBUG']:
            self.debug = kwargs['DEBUG']['attn']

    def get_layers(self):
        return [self.qkv_layer]

    def forward(self, x):
        if self.debug : self.start = perf_counter()

        B, T, _ = x.shape

        self.qkv_layer.forward(x)                                       # (B, T, 3*D)

        # tmp = 
        Q, K, V = np.split((self.qkv_layer.output.reshape((B, T, self.n_heads, 3*self.embd_dim // self.n_heads))).transpose(0,2,1,3), 3, axis=-1)

        # Q, K, V = np.split(self.qkv_layer.output, 3, axis=-1)
        self.Q, self.K, self.V = Q, K, V                                # save for backward - each (B, H, T, Dh)

        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) * self.scale       # (B, H, T, T)
        scores = scores + self.mask[:, :, :T, :T]                          # causal mask (1, 1, T, T)

        self.softmax.forward(scores)
        self.attn_weights = self.softmax.output                         # (B, H, T, T)

        attention = np.matmul(self.attn_weights, V)                     # (B, H, T, Dh)

        attention = (attention.transpose(0,2,1,3)).reshape((B, T, self.embd_dim)) # 
        if self.debug : print(f"ATTN - Forward time : {perf_counter() - self.start}")

        return attention

    def backward(self, dvalues):
        if self.debug : self.start = perf_counter()
        B, T, _ = dvalues.shape

        dvalues = (dvalues.reshape((B, T, self.n_heads, self.embd_dim // self.n_heads))).transpose(0,2,1,3) # (B, T, D) -> (B, T, H, Dh) -> (B, H, T, Dh)

        # Gradient w.r.t V
        # attn_out = softmax_weights @ V  →  dV = softmax_weights.T @ d_attn_out
        dV = np.matmul(self.attn_weights.transpose(0, 1, 3, 2), dvalues)  # (B, H, T, Dh)

        # Gradient w.r.t attention weights
        # d_attn_weights = d_attn_out @ V.T
        d_attn_weights = np.matmul(dvalues, self.V.transpose(0, 1, 3, 2)) # (B, H, T, T)

        # Backprop through softmax
        self.softmax.backward(d_attn_weights)
        d_scores = self.softmax.dinputs                                  # (B, H, T, T)

        # Zero out gradients at masked positions
        d_scores *= (self.mask[:, :, :T, :T] > -1e8)

        # Gradients w.r.t Q and K
        # scores = Q @ K.T * scale  →  dQ = d_scores @ K * scale
        #                               dK = d_scores.T @ Q * scale
        dQ = np.matmul(d_scores, self.K) * self.scale                   # (B, H, T, Dh)
        dK = np.matmul(d_scores.transpose(0, 1, 3, 2 ), self.Q) * self.scale # (B, H, T, Dh)

        # Concatenate dQ, dK, dV back too single matrix
        d_qkv = np.concatenate([dQ, dK, dV], axis=-1)                  # (B, H, T, 3*Dh)

        d_qkv = (d_qkv.transpose(0,2,1,3)).reshape((B, T, 3*self.embd_dim))

        # Backprop through fused QKV layer
        self.qkv_layer.backward(d_qkv)

        if self.debug : print(f"ATTN : Backward time : {perf_counter() - self.start}")

        return self.qkv_layer.dinputs                                    # (B, T, D)