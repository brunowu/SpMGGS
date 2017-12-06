#!/bin/bash
#SBATCH --comment "Hello ROMEO!"
#SBATCH -J "TEST 1"

#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

#SBATCH --time=02:30:00

#SBATCH -n 1
#SBATCH -N 1
#SBATCH --gres=gpu:1

echo $CUDA_VISIBLE_DEVICES

#srun -n 2 ./matgen.exe -dim 10000 -d1 8 -degree 4 -mat_type aijcusparse -vec_type cuda

./test.exe

