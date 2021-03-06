#!/bin/bash
#SBATCH --comment "Hello ROMEO!"
#SBATCH -J "TEST 1"

#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

#SBATCH --time=02:30:00

#SBATCH -n 4
#SBATCH -N 4
#SBATCH --gres=gpu:1

echo $CUDA_VISIBLE_DEVICES

#srun -n 2 ./matgen.exe -dim 10000 -d1 8 -degree 4 -mat_type aijcusparse -vec_type cuda


#mpirun -np 2 ./test.exe -log_view

mpirun -np 4 ./test.exe -dim 40000 -d1 8 -degree 4 -log_view
