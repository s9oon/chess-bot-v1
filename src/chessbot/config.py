"""Every setting you might want to tweak, in one place."""
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent  # repo root (this file is in src/chessbot/)

# --- Data ---
DATA_DIR = ROOT / "data"
EVALS_FILE = DATA_DIR / "evals_10M.jsonl"  # Lichess positions with Stockfish evaluations
NUM_POSITIONS = 3_000_000                  # how many positions to load and train on
CP_SCALE = 400                             # targets = tanh(cp / CP_SCALE): 400 cp (4 pawns) -> 0.76
WEIGHTS_DIR = DATA_DIR / "weights"         # saved models (.npz)

# --- Model ---
CHANNELS = 64  # kernels per conv layer = output feature maps per layer
HIDDEN = 64    # numbers in the head's hidden linear layer

# --- Training ---
LEARNING_RATE = 0 #Not sure what to set to
BATCH_SIZE = 1024  # positions per train_step