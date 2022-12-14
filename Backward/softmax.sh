#!/usr/bin/env zsh
#SBATCH --job-name=Softmax
#SBATCH --partition=wacc
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:05:00
#SBATCH --output=Softmax.out --error=Softmax.err


nvcc softmax_backward.cu driver_softmax.cpp -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -lcublas -std c++17 -o softmax

n=1024
dim=64
batch_size=64
num_heads=8

# nvprof --export-profile exp.txt ./gpu_baseline $n $dim
echo "Softmax Backward"

nvprof --export-profile exp.txt -f ./softmax $n $dim $batch_size $num_heads
