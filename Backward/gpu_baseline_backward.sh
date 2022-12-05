#!/usr/bin/env zsh
#SBATCH --job-name=GPU_Baseline 
#SBATCH --partition=wacc
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:00:10
#SBATCH --output=GPU_Baseline.out --error=GPU_Baseline.err


nvcc gpu_baseline_backward.cu softmax_backward.cu baseline_cpu_backward.cpp driver_backward.cpp -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -lcublas -std c++17 -o gpu_baseline_backward

n=4
dim=3 

# nvprof --export-profile exp.txt ./gpu_baseline $n $dim

./gpu_baseline_backward $n $dim
