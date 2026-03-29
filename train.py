"""
RawFormer — Training Script
Run: python train.py
"""

import math
import cupy as np

from config import (
    TRAIN_PATH, TEST_PATH, VALID_PATH,
    TRAIN_TOKENS, VAL_TOKENS, TEST_TOKENS,
    EMBD_DIM, NUM_LAYERS, N_HEADS, CONTEXT,
    EPOCHS, BATCH_SIZE, LEARNING_RATE, WARMUP_STEPS,
    VAL_EVERY, PATIENCE, CHECKPOINT_DIR, CHECKPOINT_NAME
)
from data.dataloader   import load_ptb, flatten, create_windows
from rawformer.decoder import Decoder 
from rawformer.loss import Loss_CrossCategoricalEntropy
from rawformer.optimizer import OptimizerAdam

from checkpoint import save_model


def train():
    # ------------------------------------------------------------------ #
    # 1. Data
    # ------------------------------------------------------------------ #
    print("Loading data...")
    train_data = load_ptb(TRAIN_PATH)
    valid_data = load_ptb(VALID_PATH)

    train_stream_og = flatten(train_data)

    train_stream = train_stream_og[:TRAIN_TOKENS]
    vocab_stream = train_stream_og[:TRAIN_TOKENS + VAL_TOKENS + TEST_TOKENS]
    val_stream   = train_stream_og[TRAIN_TOKENS : TRAIN_TOKENS + VAL_TOKENS]
    # ------------------------------------------------------------------ #
    # 2. Model
    # ------------------------------------------------------------------ #
    print("Building model...")
    model = Decoder(
        corpus     = [vocab_stream],
        n_heads    = N_HEADS,
        num_layers = NUM_LAYERS,
        embd_dim   = EMBD_DIM,
        context    = CONTEXT,
        tokenise   = False,
    )

    train_ids = np.array([model.vocab[w] for w in train_stream])
    val_ids   = np.array([model.vocab[w] for w in val_stream])

    X_train, Y_train = create_windows(train_ids, CONTEXT)
    X_val,   Y_val   = create_windows(val_ids,   CONTEXT)

    print(f"  Train windows : {len(X_train):,}")
    print(f"  Val windows   : {len(X_val):,}")
    print(f"  Vocab size    : {model.vocab_size:,}")

    # ------------------------------------------------------------------ #
    # 3. Training objects
    # ------------------------------------------------------------------ #
    loss_fn   = Loss_CrossCategoricalEntropy()
    optimizer = OptimizerAdam(
        learning_rate = LEARNING_RATE,
        warmup_steps  = WARMUP_STEPS,
    )
    layers = model.get_all_layers()   # built once, reused every epoch

    best_val_ppl     = float('inf')
    patience_counter = 0

    # ------------------------------------------------------------------ #
    # 4. Epoch loop
    # ------------------------------------------------------------------ #
    for epoch in range(EPOCHS):

        # Shuffle
        indices = np.random.permutation(len(X_train))
        X_train = X_train[indices]
        Y_train = Y_train[indices]

        epoch_loss  = 0.0
        num_batches = len(X_train) // BATCH_SIZE

        # ---- Batch loop ----
        for b in range(0, len(X_train), BATCH_SIZE):
            X_batch = X_train[b : b + BATCH_SIZE]
            Y_batch = Y_train[b : b + BATCH_SIZE]

            # Zero embedding grad buffer before each batch
            model.dembeddings.fill(0)

            logits = model.forward(X_batch)
            loss   = loss_fn.calculate(logits, Y_batch)
            loss_fn.backward(logits, Y_batch)
            model.backward(loss_fn.dinputs)

            optimizer.pre_update()
            for layer in layers:
                optimizer.update_params(layer)
            optimizer.update_params_embeddings(model)

            # Keep LM head weights tied to embeddings after each update
            model.lm_head.weights = model.embeddings.T

            epoch_loss += loss

        train_loss = (epoch_loss / num_batches).get()

        # ---- Validation ----
        if epoch % VAL_EVERY == 0:
            val_loss    = 0.0
            val_batches = 0

            for b in range(0, len(X_val), BATCH_SIZE):
                X_batch  = X_val[b : b + BATCH_SIZE]
                Y_batch  = Y_val[b : b + BATCH_SIZE]
                logits   = model.forward(X_batch)
                val_loss += loss_fn.calculate(logits, Y_batch)
                val_batches += 1

            mean_val_loss = (val_loss / val_batches).get()
            val_ppl       = math.exp(mean_val_loss)

            # Early stopping
            if val_ppl < best_val_ppl:
                best_val_ppl     = val_ppl
                patience_counter = 0
                save_model(model, f"{CHECKPOINT_DIR}/best_{CHECKPOINT_NAME}")
            else:
                patience_counter += 1

            print(
                f"Epoch {epoch:>3} | "
                f"train loss: {train_loss:.4f} | "
                f"val loss: {mean_val_loss:.4f} | "
                f"val ppl: {val_ppl:.2f} | "
                f"patience: {patience_counter}/{PATIENCE}"
            )

            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}. "
                      f"Best val perplexity: {best_val_ppl:.2f}")
                break
        else:
            print(f"Epoch {epoch:>3} | train loss: {train_loss:.4f}")

    print("\nTraining complete.")
    save_model(model, f"{CHECKPOINT_DIR}/{CHECKPOINT_NAME}")
    return model


if __name__ == "__main__":
    train()