import cupy as np
import cupyx
from math import sqrt
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

from rawformer.layers import Layer_Dense, LayerNorm
from rawformer.activations import Activation_Softmax
from rawformer.blocks import DecoderBlock


class Decoder:
    """
    Decoder-only transformer (GPT style) built from scratch using CuPy.

    Architecture:
        Token Embeddings + Positional Encoding
        → N x DecoderBlock (Pre-LN, causal self-attention + FFN)
        → Final LayerNorm
        → LM Head (weight-tied to embeddings)
        → Softmax
    """

    def __init__(self, corpus, n_heads, num_layers, embd_dim, context, tokenise=False):
        if tokenise:
            self.tokenized_corpus = [word_tokenize(s.lower()) for s in corpus]
        else:
            self.tokenized_corpus = corpus

        self.embd_dim  = embd_dim
        self.context   = context
        self.n_heads   = n_heads

        self.vocab_creation()

        # Embeddings (random init, scaled small)
        self.embeddings = (
            np.random.randn(self.vocab_size, self.embd_dim).astype(np.float32) * 0.02
        )

        # Positional Encoding (vectorized sinusoidal -> may experiment with RoPE later)
        self.position_encodings = self.create_positional_encoding(context)

        # Transformer Blocks
        self.blocks = [DecoderBlock(embd_dim, context) for _ in range(num_layers)]

        # Final LayerNorm
        self.final_norm = LayerNorm(embd_dim)

        # LM Head (weight-tied to embeddings)
        self.lm_head  = Layer_Dense(embd_dim, self.vocab_size)
        self.lm_head.weights = self.embeddings.T   # weight tying

        # Output Softmax
        self.final_act = Activation_Softmax()

        # Embedding gradient buffer
        self.dembeddings = np.zeros_like(self.embeddings)

    # ------------------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------------------

    def vocab_creation(self):
        self.vocab = {}
        indx = 0
        for sentence in self.tokenized_corpus:
            for word in sentence:
                if word not in self.vocab:
                    self.vocab[word] = indx
                    indx += 1
        self.vocab_size = len(self.vocab)

    # ------------------------------------------------------------------
    # Optional: Word2Vec embedding initialization
    # ------------------------------------------------------------------
    
    # NOTE: This is an optional method to initialize the embedding matrix using Word2Vec instead of random initialization. 
    # It can be called after instantiating the Decoder class if desired. 
    def word_embeddings(self):
        data = []
        for i in self.tokenized_corpus:
            temp = ['sos'] + [j.lower() for j in i]
            data.append(temp)

        model = Word2Vec(sentences=data, vector_size=512, window=5, min_count=1, workers=4)
        words = list(model.wv.index_to_key)
        self.words = words

        embeddings_matrix = np.zeros((len(words), model.vector_size), dtype=np.float32)
        for i, word in enumerate(words):
            embeddings_matrix[i] = model.wv[word]

        self.embeddings = (embeddings_matrix * sqrt(self.embd_dim)).astype(np.float32)

    # ------------------------------------------------------------------
    # Positional Encoding (vectorized sinusoidal)
    # ------------------------------------------------------------------

    def create_positional_encoding(self, context):
        pos    = np.arange(context, dtype=np.float32)[:, None]             # (T, 1)
        dims   = np.arange(0, self.embd_dim, 2, dtype=np.float32)[None, :] # (1, D/2)
        angles = pos / np.power(10000.0, dims / self.embd_dim)             # (T, D/2)

        pe = np.zeros((1, context, self.embd_dim), dtype=np.float32)
        pe[0, :, 0::2] = np.sin(angles)
        pe[0, :, 1::2] = np.cos(angles)
        return pe

    # ------------------------------------------------------------------
    # Layer registry (used by optimizer)
    # ------------------------------------------------------------------

    def get_all_layers(self):
        layers = []
        for block in self.blocks:
            layers.append(block.attn.qkv_layer)
            layers.append(block.ffn.fc1)
            layers.append(block.ffn.fc2)
            layers.append(block.norm1)
            layers.append(block.norm2)
        layers.append(self.final_norm)
        # lm_head excluded: weight-tied to embeddings, updated via embedding optimizer
        return layers

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, input_ids):
        self.last_input_ids = input_ids           # save for backward

        x = self.embeddings[input_ids]            # (B, T, D)
        x = x + self.position_encodings[:, :x.shape[1], :]

        for block in self.blocks:
            x = block.forward(x)

        self.final_norm.forward(x)
        self.lm_head.forward(self.final_norm.output)
        self.final_act.forward(self.lm_head.output)

        return self.final_act.output              # (B, T, vocab_size) — probabilities

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def backward(self, dvalues):
        self.lm_head.backward(dvalues)

        # Accumulate gradient into embedding matrix (weight tying)
        self.dembeddings += self.lm_head.dweights.T

        self.final_norm.backward(self.lm_head.dinputs)
        dvalues = self.final_norm.dinputs

        for block in reversed(self.blocks):
            dvalues = block.backward(dvalues)

        # Scatter embedding gradients (GPU-native scatter_add)
        flat_ids   = self.last_input_ids.reshape(-1)       # (B*T,)
        flat_grads = dvalues.reshape(-1, self.embd_dim)    # (B*T, D)
        cupyx.scatter_add(self.dembeddings, flat_ids, flat_grads)

        return dvalues