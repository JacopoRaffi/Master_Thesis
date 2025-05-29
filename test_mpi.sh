#!/bin/bash
#SBATCH --job-name=mpi_intel_job
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=job_output.txt
#SBATCH --cpus-per-task=56

# Run using mpiexec or mpirun (Hydra launcher)
export LD_LIBRARY_PATH=$HOME/libomp:$LD_LIBRARY_PATH
mpirun python mpi_test.py