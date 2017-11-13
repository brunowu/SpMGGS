#!/bin/bash
#SBATCH --comment "Hello ROMEO!"
#SBATCH -J "TEST 1"

#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

#SBATCH --time=100:30:00

#SBATCH -n 1

MPI_NODES=1
DIM=100
D1=3
DEGREE=1

srun -n ${MPI_NODES} ./matgen.exe -dim ${DIM} -d1 ${D1} -degree ${DEGREE}


