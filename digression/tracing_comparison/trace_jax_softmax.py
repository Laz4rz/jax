from jax.numpy import array
from jax.nn import softmax

p = array([0.50, 0.60, 0.70, 0.30, 0.25])
s = softmax(p)

print(s)
