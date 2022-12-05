#!/usr/bin/env zsh
#SBATCH --job-name=GPU_Baseline
#SBATCH --mem=8G
#SBATCH --partition=wacc
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:02:00
#SBATCH --output=GPU_Baseline.out --error=GPU_Baseline.err

nvcc gpu_baseline.cu softmax_cublas.cu baseline_cpu.cpp driver.cpp -g -G -lcublas -std c++17 -o gpu_baseline

n=64
dim=64 
# -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 

nvprof --export-profile exp.txt ./gpu_baseline $n $dim



# itr=13

# for ((i=6; i<=itr; i++)); do
#     echo "Running with n = $n, dim = $dim"
#     ./driver_cpu $n $dim
#     n=$((n*2))
# done
