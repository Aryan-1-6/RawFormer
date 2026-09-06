import cupy as np
from rawformer import LayerNorm, SelfAttention, FeedForward
from time import perf_counter

class DecoderBlock:
    """
    Single transformer decoder block using Pre-LN (GPT-2 style):

        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))

    Pre-LN places LayerNorm before each sub-layer, which stabilizes
    training without requiring careful learning rate warmup.
    """

    def __init__(self, embd_dim, context, n_heads, **kwargs):
        self.norm1 = LayerNorm(embd_dim)
        self.attn  = SelfAttention(embd_dim, context, n_heads, DEBUG=kwargs['DEBUG'])
        self.norm2 = LayerNorm(embd_dim)
        self.ffn   = FeedForward(embd_dim, DEBUG=kwargs['DEBUG'])
        self.num_block = kwargs['num_block']
        self.start = 0

        self.debug = False        
        if kwargs['DEBUG']:
            self.debug = kwargs['DEBUG']['block']

    def forward(self, x):
        if self.debug : 
            np.cuda.Stream.null.synchronize()
            self.start = perf_counter()

        # Attention sub-layer (Pre-LN)
        residual = x
        x = self.norm1.forward(x)
        attn_out = self.attn.forward(x)
        x = residual + attn_out          # residual on original x, not normalized

        # FFN sub-layer (Pre-LN)
        residual = x
        x = self.norm2.forward(x)
        ffn_out  = self.ffn.forward(x)
        x = residual + ffn_out

        if self.debug : 
            np.cuda.Stream.null.synchronize()
            print(f"DECODER Block {self.num_block} - Forward time : {perf_counter() - self.start}")

        return x

    def backward(self, dvalues):
        if self.debug : 
            np.cuda.Stream.null.synchronize()
            self.start = perf_counter()

        # FFN residual branch
        # Gradient flows through both: skip path (dvalues) + FFN path (d_ffn_branch)
        d_ffn_branch = self.ffn.backward(dvalues)
        d_ffn_branch = self.norm2.backward(d_ffn_branch)
        dvalues = dvalues + d_ffn_branch

        # Attention residual branch
        d_attn_branch = self.attn.backward(dvalues)
        d_attn_branch = self.norm1.backward(d_attn_branch)
        dvalues = dvalues + d_attn_branch

        if self.debug : 
            np.cuda.Stream.null.synchronize()
            print(f"DECODER Block {self.num_block} - Backward time : {perf_counter() - self.start}")

        return dvalues