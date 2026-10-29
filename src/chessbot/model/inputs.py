"""Packed boards (engine/encoding.py) -> the network's input planes, in JAX on the GPU."""
import jax
import jax.numpy as jnp


@jax.jit
def unpack_planes(packed: jax.Array) -> jax.Array:
    """Expand packed boards into planes on the GPU, any number of rows: (n, 98) -> (n, 8, 8, 18)."""
    n = packed.shape[0]

    # 96 bytes -> 768 bits in square order -> [plane, rank, file] -> [rank, file, plane].
    pieces = jnp.unpackbits(packed[:, :96], axis=1, bitorder="little")
    pieces = pieces.reshape(n, 12, 8, 8).transpose(0, 2, 3, 1)

    # Each flag bit fills a whole plane.
    flags = (packed[:, 96:97] >> jnp.arange(5, dtype=jnp.uint8)) & 1
    flags = jnp.broadcast_to(flags[:, None, None, :], (n, 8, 8, 5))

    # One-hot over 65 slots, then drop slot 64 ("none") so no en passant = all zeros.
    ep = jax.nn.one_hot(packed[:, 97], 65)[:, :64].reshape(n, 8, 8, 1)

    return jnp.concatenate([pieces, flags, ep], axis=-1).astype(jnp.float32)
