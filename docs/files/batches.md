# batches.py

[`src/batches.py`](../../src/batches.py): shared by every training method.

```python
import numpy as np


def batches(boards: np.ndarray, targets: np.ndarray, batch_size: int, rng: np.random.Generator):
    order = rng.permutation(len(boards))
    for start in range(0, len(order) - batch_size + 1, batch_size):
        idx = order[start:start + batch_size]
        yield boards[idx], targets[idx]
```

**Does:** shuffles the row order once, then hands out `(boards, targets)` in
chunks of `batch_size` rows, one at a time, still packed. The last few rows
that don't fill a whole chunk are skipped. Calling it again gives a new shuffle,
so different rows get skipped each time.

**Used:** by the training loop, called once per epoch; each batch then goes
through [`unpack_planes`](encoding.md#unpacking-performed-by-the-gpu) and
`train_step`. Nothing calls it yet: the loop in [supervised.py](supervised.md)
isn't written.
