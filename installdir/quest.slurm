#!/bin/bash

#SBATCH --job-name=32QubitQFT
#SBATCH --time=00:20:00
#SBATCH --exclusive
#SBATCH --nodes=2
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --nodelist=cn03,cn04

/usr/sbin/slurmd -C
module load PrgEnv/arm-20.0/openmpi-4.0.2 

# Change to the submission directory
cd $SLURM_SUBMIT_DIR

# Set the number of thread
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Set run parameters
NUM_QUBITS=32

# Report run information
echo "COMPILED AT -O3"
echo "QFT ON $NUM_QUBITS QUBITS"
echo "SLURM_JOB_NUM_NODES=$SLURM_JOB_NUM_NODES"
echo "SLURM_TASKS_PER_NODE=$SLURM_TASKS_PER_NODE"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"

# Launch the parallel job
mpirun -N 1 -n ${SLURM_JOB_NUM_NODES} --bind-to core --map-by ppr:1:node:pe=${SLURM_CPUS_PER_TASK} --display-map ./bins/qft ${NUM_QUBITS}

# Report job time
echo ""
echo "ROUGH WALLTIME: (mm:ss)"
squeue -h -j $SLURM_JOBID -o "%M"
