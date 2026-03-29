import cupy as np


def load_ptb(path):
    """
    Load a PTB-format text file.
    Each line becomes a token list with 'sos' prepended and 'eos' appended.
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = f.readlines()

    tokenized = []
    for line in data:
        tokens = ['sos'] + line.strip().split() + ['eos']
        tokenized.append(tokens)

    return tokenized


def flatten(data):
    """Flatten a list of token lists into a single token stream."""
    stream = []
    for sent in data:
        stream.extend(sent)
    return stream


def create_windows(token_ids, context):
    """
    Slide a context window over a token ID stream to produce
    (input, target) pairs for language model training.

    Args:
        token_ids : 1-D CuPy array of integer token IDs
        context   : window size (sequence length)

    Returns:
        X : (N, context) int array — input windows
        Y : (N, context) int array — target windows (shifted by 1)
    """
    X, Y = [], []
    for i in range(len(token_ids) - context):
        X.append(token_ids[i     : i + context])
        Y.append(token_ids[i + 1 : i + context + 1])
    return np.array(X), np.array(Y)


def create_lm_pairs(sentences):
    """
    Create (input, target) sentence pairs for language modelling.
    Input = sentence[:-1], Target = sentence[1:].
    Note: create_windows() is preferred for PTB-style training.
    """
    X, Y = [], []
    for sent in sentences:
        if len(sent) < 2:
            continue
        X.append(sent[:-1])
        Y.append(sent[1:])
    return X, Y