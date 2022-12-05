#!/usr/bin/env zsh
#SBATCH --job-name=CPU_Baseline
#SBATCH --mem=8G
#SBATCH --partition=wacc
#SBATCH -c 1
#SBATCH --time=0-00:02:00
#SBATCH --output=CPU_Baseline.out --error=CPU_Baseline.err

g++ baseline_cpu.cpp driver.cpp -Wall -O3 -std=c++17 -o driver_cpu

n=64
dim=64

itr=13

for ((i=6; i<=itr; i++)); do
    echo "Running with n = $n, dim = $dim"
    ./driver_cpu $n $dim
    n=$((n*2))
done
