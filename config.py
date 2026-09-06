# =============================================================================
# RawFormer — Configuration
# All hyperparameters live here. Edit this file before running train.py.
# =============================================================================

import ml_dtypes

# ---- Data ----
TRAIN_PATH = 'data/ptbdataset/ptb.train.txt'
TEST_PATH  = 'data/ptbdataset/ptb.test.txt'
VALID_PATH = 'data/ptbdataset/ptb.valid.txt'

# How many tokens from the flat stream to use for each split
TRAIN_TOKENS =   64000
VAL_TOKENS   =   18000    # tokens after TRAIN_TOKENS
TEST_TOKENS  =   18000    # tokens after VAL_TOKENS

# ---- Model ----
EMBD_DIM   = 512     # embedding / hidden dimension
NUM_LAYERS = 4       # number of transformer blocks
N_HEADS    = 4       # number of attention heads
CONTEXT    = 128     # sequence length / context window

# ---- Training ----
EPOCHS       = 10
BATCH_SIZE   = 64
LEARNING_RATE = 0.0003
WARMUP_STEPS  = 200    # linear LR warmup steps

# ---- Early Stopping ----   NOTE : Early Stopping is disabled for now
VAL_EVERY  = 1        # validate every N Epochs
PATIENCE   = 5        # stop after this many val checks with no improvement

# ---- Define Data-type ---- 
DTYPE = ml_dtypes.bfloat16    # Used for faster lower precision training instead of np.float32/np.float64

# ---- Checkpointing ----
CHECKPOINT_DIR  = 'checkpoints'
CHECKPOINT_NAME = 'flagship_rawformer.pkl'
CHECKPOINT_FLAG = False        # Keep True if you want to continue training from latest model checkpoint

DEBUG = False
DEBUG_OPTIONS = {
    "attn" : False,
    "ffn" : False,
    "block" : False,
    "total": False
}