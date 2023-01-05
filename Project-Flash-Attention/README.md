## Forward
In Forward -> `sbatch cpu_baseline.sh`, `sbatch gpu_baseline.sh`, 

## Softmax
In Softmax --> `sbatch softmax.sh`

## Backward

### Softmax
Run `sbatch softmax.sh` by setting the appropriate n(sequence length), dim, batch_size and num_heads

The header `self_attention_backward.cuh` contains various flavors of softmax grad (softmax_backward(1,2,3)) in order of increasing optimization.

### Backward Pass

Run `gpu_baseline_backward.sh` by setting the appropriate n(sequence length), dim, batch_size and num_heads
Set the softmax version to be used in the `softmax_backward` function in `softmax_backward.cu` file.