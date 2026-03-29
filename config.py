# =============================================================================
# RawFormer — Configuration
# All hyperparameters live here. Edit this file before running train.py.
# =============================================================================

# ---- Data ----
TRAIN_PATH = 'data/ptbdataset/ptb.train.txt'
TEST_PATH  = 'data/ptbdataset/ptb.test.txt'
VALID_PATH = 'data/ptbdataset/ptb.valid.txt'

# How many tokens from the flat stream to use for each split
TRAIN_TOKENS = 1200
VAL_TOKENS   =   300    # tokens after TRAIN_TOKENS
TEST_TOKENS  =   300    # tokens after VAL_TOKENS

# ---- Model ----
EMBD_DIM   = 256     # embedding / hidden dimension
NUM_LAYERS = 4       # number of transformer blocks
N_HEADS    = 4       # number of attention heads (informational — MHA not yet implemented)
CONTEXT    = 128     # sequence length / context window

# ---- Training ----
EPOCHS       = 50
BATCH_SIZE   = 64
LEARNING_RATE = 0.0003
WARMUP_STEPS  = 200    # linear LR warmup steps

# ---- Early Stopping ----
VAL_EVERY  = 1        # validate every N epochs
PATIENCE   = 5        # stop after this many val checks with no improvement

# ---- Checkpointing ----
CHECKPOINT_DIR  = 'checkpoints'
CHECKPOINT_NAME = 'rawformer.pkl'