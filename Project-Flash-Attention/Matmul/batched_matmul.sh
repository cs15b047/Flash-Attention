#!/usr/bin/env zsh
#SBATCH --job-name=batched_matmul
#SBATCH --mem=8G
#SBATCH --partition=wacc
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:02:00
#SBATCH --output=batched_matmul.out --error=batched_matmul.err

module load nvidia/cuda
nvcc driver_batch_matmul.cpp batch_matmul_baseline.cu  -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -lcublas -std c++17 -o batched_matmul

B=2
M=1024
N=1024
K=64

./batched_matmul $B $M $N $K