with open("trace_torch_softmax.log", "r") as file:
    trace_torch_aoftmax = file.read()

with open("trace_jax_softmax.log", "r") as file:
    trace_jax_softmax = file.read()

print(f"Torch total calls: {trace_torch_aoftmax.count('->')}")
print(f"JAX total calls: {trace_jax_softmax.count('->')}")
