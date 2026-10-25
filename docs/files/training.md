# training.py

[`src/training.py`](../../src/training.py). The training loop isn't written
yet; one step is:

```python
@nnx.jit
def train_step(model, optimizer, boards, targets):
    loss, grads = nnx.value_and_grad(loss_fn)(model, boards, targets)
    optimizer.update(model, grads)
    return loss
```

- `loss_fn` is mean squared error between the model's scores and the targets.
- `value_and_grad` gives the loss and a gradient for every weight
  (backpropagation).
- `optimizer.update` is Adam nudging each weight against its gradient. It only
  updates `nnx.Param`, not batchnorm's running averages
  ([train vs eval](../concepts/batch_normalization.md#train-mode-vs-eval-mode)).
