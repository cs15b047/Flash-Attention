#!/usr/bin/env zsh
#SBATCH --job-name=Task6
#SBATCH --partition=wacc
#SBATCH -c 2
#SBATCH --time=0-00:00:10
#SBATCH --output=Task6.out --error=Task6.err

g++ task6.cpp -Wall -O3 -std=c++17 -o task6
./task6 10