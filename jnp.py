from timeit import timeit

import jax
import jax.numpy as jnp
from jax import random

print(f"JAX uses {jax.devices()[0]}")
print(f"All devices: {jax.devices()}")


n_repetitions = 3
random_key = random.key(0)
n = m = 10000


a = random.normal(random_key, (n, m))

duration_without_blocking = timeit(lambda: jnp.dot(a, a.T), number=n_repetitions)
duration_with_blocking = timeit(lambda: jnp.dot(a, a.T).block_until_ready(), number=n_repetitions)

print(f"JAX .dot on {m, n} without blocking: {duration_without_blocking:.2f} seconds")
print(f"JAX .dot on {m, n} with blocking: {duration_with_blocking:.2f} seconds")
