#!/usr/bin/env zsh
#SBATCH --job-name=GPU_Baseline 
#SBATCH --partition=wacc
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:05:00
#SBATCH --output=GPU_Baseline.out --error=GPU_Baseline.err


nvcc gpu_baseline_backward.cu softmax_backward.cu baseline_cpu_backward.cpp driver_backward.cpp -Xcompiler \
 -O3 -Xcompiler -Wall -Xptxas -O3 -lcublas -std c++17 \
  -o gpu_baseline_backward

n=1024
dim=64
batch_size=64
num_heads=8

# nvprof --export-profile exp.txt ./gpu_baseline $n $dim
echo "Executing GPU Baseline Backward"

./gpu_baseline_backward $n $dim $batch_size $num_heads
