#!/usr/bin/env zsh
#SBATCH --job-name=Matmul_Baseline
#SBATCH --mem=8G
#SBATCH --partition=wacc
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:02:00
#SBATCH --output=Matmul_Baseline.out --error=Matmul_Baseline.err

module load nvidia/cuda
nvcc matmul_baseline.cu driver_matmul.cpp -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -lcublas -std c++17 -o matmul_baseline

M=4096
N=4096
K=256

./matmul_baseline $M $N $K
