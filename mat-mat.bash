#!/bin/bash
#PJM -L rscgrp=lecture-flat
#PJM -L node=16
#PJM --mpi proc=1024
#PJM -L elapse=00:01:00
#PJM -g gt05

#export I_MPI_DEBUG=4
export OMP_NUM_THREADS=4
mpiexec.hydra -n ${PJM_MPI_PROC} ./mat-mat

source /usr/local/bin/mpi_core_setting.sh

mpiexec.hydra -n ${PJM_MPI_PROC} ./mat-mat

export I_MPI_HBW_POLICY=hbw_preferred

mpiexec.hydra -n ${PJM_MPI_PROC} ./mat-mat
