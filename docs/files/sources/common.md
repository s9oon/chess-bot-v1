# sources/common.py

[`src/sources/common.py`](../../../src/sources/common.py): the parts every
source shares.

## `batches(boards, targets, batch_size, rng)`

One epoch of shuffled batches, still packed. You give it everything (all `N`
rows in RAM) plus the batch size, and it hands out `batch_size` rows at a time.
Training sends each batch to the GPU and unpacks it with
[`unpack_planes`](../encoding.md#unpack_planespacked---jaxarray).

The last few rows that don't fill a whole batch are skipped that epoch. The
shuffle is different every epoch, so different rows get skipped each time.
