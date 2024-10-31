#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4000
#SBATCH --time=48:00:00

module purge

module load GCCcore/8.3.0
module load Python/3.7.4

module load GCC/11.2.0  OpenMPI/4.1.1
module load SciPy-bundle/2021.10

module load GCC/10.2.0  OpenMPI/4.0.5
module load numba/0.52.0


MY_NUM_THREADS=$SLURM_CPUS_PER_TASK

export OMP_NUM_THREADS=$MY_NUM_THREADS

python 9_02_taskfarm_401x401_100N_seed42v_eqspace_0_400da_50K.py