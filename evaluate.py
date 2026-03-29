"""
RawFormer — Evaluation Script
Computes perplexity on the test set.

Run: python evaluate.py
"""

import math
import cupy as np

from config import TEST_TOKENS, TRAIN_TOKENS, VAL_TOKENS, TEST_PATH, TRAIN_PATH, CONTEXT, BATCH_SIZE, CHECKPOINT_DIR, CHECKPOINT_NAME
from data.dataloader import load_ptb, flatten, create_windows
from rawformer import Decoder 
from checkpoint import load_model


def test_perplexity(model, X_test, Y_test, batch_size=64):
    """
    Compute perplexity of the model on test windows.

    Perplexity = exp(mean cross-entropy loss over all tokens)
    Lower = better. Random-guess baseline ≈ vocab_size.
    """
    total_log_loss = 0.0
    total_tokens   = 0
    num_batches    = 0

    for b in range(0, len(X_test), batch_size):
        X_batch = X_test[b : b + batch_size]
        Y_batch = Y_test[b : b + batch_size]

        logits = model.forward(X_batch)          # (B, T, V) — softmax probs
        B, T, V = logits.shape

        probs_flat   = logits.reshape(-1, V)
        targets_flat = Y_batch.reshape(-1)

        correct_probs = probs_flat[np.arange(len(targets_flat)), targets_flat]
        nll = -np.log(correct_probs + 1e-9)

        total_log_loss += float(nll.sum())
        total_tokens   += len(targets_flat)
        num_batches    += 1

        if num_batches % 20 == 0:
            running_ppl = math.exp(total_log_loss / total_tokens)
            print(f"  Batch {num_batches} | running perplexity: {running_ppl:.2f}")

    mean_nll   = total_log_loss / total_tokens
    perplexity = math.exp(mean_nll)

    print(f"\n=== Test Results ===")
    print(f"  Tokens evaluated : {total_tokens:,}")
    print(f"  Mean NLL         : {mean_nll:.4f}")
    print(f"  Perplexity       : {perplexity:.2f}")
    print(f"  Vocab size       : {model.vocab_size}  "
          f"(random-guess baseline ≈ {model.vocab_size})")

    return perplexity


def evaluate():
    # ---- Load vocab from train data ----
    train_data   = load_ptb(TRAIN_PATH)
    vocab_stream = flatten(train_data)[:TRAIN_TOKENS + VAL_TOKENS + TEST_TOKENS]

    # test_data   = load_ptb(TEST_PATH)
    test_stream = vocab_stream[TRAIN_TOKENS + VAL_TOKENS:TRAIN_TOKENS + VAL_TOKENS + TEST_TOKENS]

    # ---- Rebuild model shell and load weights ----
    model = Decoder(
        corpus     = [vocab_stream],
        n_heads    = 4,
        num_layers = 4,
        embd_dim   = 256,
        context    = CONTEXT,
        tokenise   = False,
    )
    model = load_model(f"{CHECKPOINT_DIR}/best_{CHECKPOINT_NAME}", model)

    # ---- Build test windows (handle OOV with <unk> if present) ----
    unk_id   = model.vocab.get('<unk>', 0)
    test_ids = np.array([model.vocab.get(w, unk_id) for w in test_stream])
    X_test, Y_test = create_windows(test_ids, CONTEXT)

    test_perplexity(model, X_test, Y_test, batch_size=64)


if __name__ == "__main__":
    evaluate()