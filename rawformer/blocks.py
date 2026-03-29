import cupy as np
from rawformer.layers import LayerNorm
from rawformer.attention import SelfAttention
from rawformer.feedforward import FeedForward


class DecoderBlock:
    """
    Single transformer decoder block using Pre-LN (GPT-2 style):

        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))

    Pre-LN places LayerNorm before each sub-layer, which stabilizes
    training without requiring careful learning rate warmup.
    """

    def __init__(self, embd_dim, context):
        self.norm1 = LayerNorm(embd_dim)
        self.attn  = SelfAttention(embd_dim, context)
        self.norm2 = LayerNorm(embd_dim)
        self.ffn   = FeedForward(embd_dim)

    def forward(self, x):
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

        return x

    def backward(self, dvalues):
        # FFN residual branch
        # Gradient flows through both: skip path (dvalues) + FFN path (d_ffn_branch)
        d_ffn_branch = self.ffn.backward(dvalues)
        d_ffn_branch = self.norm2.backward(d_ffn_branch)
        dvalues = dvalues + d_ffn_branch

        # Attention residual branch
        d_attn_branch = self.attn.backward(dvalues)
        d_attn_branch = self.norm1.backward(d_attn_branch)
        dvalues = dvalues + d_attn_branch

        return dvalues