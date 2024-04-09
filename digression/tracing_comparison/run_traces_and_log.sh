source /Users/mikolajboronski/miniconda3/etc/profile.d/conda.sh

echo "Running traces and logging to files"

echo "Running trace for JAX softmax"
conda activate jax
python -m trace -T trace_jax_softmax.py > trace_jax_softmax.log

echo "Running trace for PyTorch softmax"
conda activate mnlp
python -m trace -T trace_torch_softmax.py > trace_torch_softmax.log
