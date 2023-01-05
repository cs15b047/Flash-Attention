#!/usr/bin/env zsh
#SBATCH --job-name=FirstSlurm
#SBATCH --partition=wacc
#SBATCH -c 2
#SBATCH --time=0-00:00:10
#SBATCH --output=FirstSlurm.out --error=FirstSlurm.err

hostname