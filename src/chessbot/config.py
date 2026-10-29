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

# --- Search ---
NUM_SIMULATIONS = 100  # MCTS simulations per move: more = stronger and slower
C_PUCT = 1.25          # how much search trusts the policy's priors vs the values it has found (AlphaZero's c_init)
DIRICHLET_ALPHA = 0.3  # self-play root noise: AlphaZero's value for chess (smaller = noise on fewer moves)
NOISE_FRACTION = 0.25  # self-play root priors = 75% policy + 25% noise

# --- Training ---
LEARNING_RATE = 0.001  # Adam's usual starting step size
BATCH_SIZE = 1024      # positions per train_step

# --- Self-play ---
NUM_ROUNDS = 100          # each round: play some games, then train on the buffer
GAMES_PER_ROUND = 5
STEPS_PER_ROUND = 20      # train_steps per round
BUFFER_SIZE = 20_000      # most recent positions kept to train on (each pi is 4096 float32s: 16 KB, ~330 MB total)
TEMPERATURE_PLIES = 30    # first 30 plies: pick moves in proportion to visits; after: the most visited
MAX_GAME_PLIES = 300      # stop a game here and call it a draw, so early random play can't go on for ever