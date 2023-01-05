#!/usr/bin/env zsh
#!/usr/bin/env zsh
#SBATCH --job-name=GPU_Baseline
#SBATCH --mem=8G
#SBATCH --partition=wacc
#SBATCH --gres=gpu:p100:1
#SBATCH --time=0-00:02:00
#SBATCH --output=GPU_Baseline.out --error=GPU_Baseline.err

nvidia-smi
nvcc softmax.cu baseline_cpu_softmax.cpp driver.cpp -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -lcublas -std c++17 -o gpu_baseline_softmax

# bs=2
# h=4
# seq_len=5
#32 * 12 * 128 -> bert batch size * seq len * hidden size
./gpu_baseline_softmax  49152 128
#./gpu_baseline_softmax 49152 4

# itr=13

# for ((i=6; i<=itr; i++)); do
#     echo "Running with n = $n, dim = $dim"
#     ./driver_cpu $n $dim
#     n=$((n*2))
# done
