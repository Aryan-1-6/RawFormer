"""
RawFormer — Checkpoint Utilities
Saves and loads all learnable parameters to/from a .pkl file.
CuPy arrays are converted to NumPy on save (portable) and back to CuPy on load.
"""

import pickle
import os
import cupy as cp


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------

def _to_numpy(arr):
    """Convert CuPy array to NumPy. Pass-through if already NumPy."""
    return arr.get() if hasattr(arr, 'get') else arr


def _to_cupy(arr):
    """Convert NumPy array to CuPy float32."""
    return cp.array(arr, dtype=cp.float32)


def _save_dense(layer):
    return {
        'weights': _to_numpy(layer.weights),
        'biases':  _to_numpy(layer.biases),
    }


def _save_layernorm(layer):
    return {
        'gamma': _to_numpy(layer.gamma),
        'beta':  _to_numpy(layer.beta),
    }


def _save_block(block):
    return {
        'norm1': _save_layernorm(block.norm1),
        'attn':  {'qkv_layer': _save_dense(block.attn.qkv_layer)},
        'norm2': _save_layernorm(block.norm2),
        'ffn':   {
            'fc1': _save_dense(block.ffn.fc1),
            'fc2': _save_dense(block.ffn.fc2),
        },
    }


def _load_dense(layer, data):
    layer.weights = _to_cupy(data['weights'])
    layer.biases  = _to_cupy(data['biases'])


def _load_layernorm(layer, data):
    layer.gamma = _to_cupy(data['gamma'])
    layer.beta  = _to_cupy(data['beta'])


def _load_block(block, data):
    _load_layernorm(block.norm1, data['norm1'])
    _load_dense(block.attn.qkv_layer, data['attn']['qkv_layer'])
    _load_layernorm(block.norm2, data['norm2'])
    _load_dense(block.ffn.fc1, data['ffn']['fc1'])
    _load_dense(block.ffn.fc2, data['ffn']['fc2'])


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def save_model(decoder, filepath='checkpoints/rawformer.pkl'):
    """
    Serialize all learnable parameters to a pickle file.

    Args:
        decoder  : trained Decoder instance
        filepath : destination path (directories are created if needed)
    """
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

    model_data = {
        'config': {
            'embd_dim':   decoder.embd_dim,
            'context':    decoder.context,
            'vocab_size': decoder.vocab_size,
            'num_layers': len(decoder.blocks),
            'n_heads':    decoder.n_heads,
        },
        'vocab':      decoder.vocab,
        'embeddings': _to_numpy(decoder.embeddings),
        'blocks':     [_save_block(b) for b in decoder.blocks],
        'final_norm': _save_layernorm(decoder.final_norm),
        'lm_head':    _save_dense(decoder.lm_head),
    }

    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Checkpoint saved → '{filepath}'")
    print(f"  Blocks      : {len(decoder.blocks)}")
    print(f"  Vocab size  : {decoder.vocab_size:,}")
    print(f"  Embd dim    : {decoder.embd_dim}")
    print(f"  Context len : {decoder.context}")


def load_model(filepath, decoder):
    """
    Load parameters from a checkpoint into an existing Decoder instance.
    Automatically moves all arrays back to GPU (CuPy).

    Args:
        filepath : path to saved .pkl checkpoint
        decoder  : Decoder instance with matching architecture

    Returns:
        decoder with loaded weights
    """
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)

    decoder.vocab      = model_data['vocab']
    decoder.vocab_size = model_data['config']['vocab_size']
    decoder.embeddings = _to_cupy(model_data['embeddings'])

    for block, block_data in zip(decoder.blocks, model_data['blocks']):
        _load_block(block, block_data)

    _load_layernorm(decoder.final_norm, model_data['final_norm'])
    _load_dense(decoder.lm_head, model_data['lm_head'])

    # Re-apply weight tying after loading
    decoder.lm_head.weights = decoder.embeddings.T

    print(f"Checkpoint loaded ← '{filepath}'")
    return decoder