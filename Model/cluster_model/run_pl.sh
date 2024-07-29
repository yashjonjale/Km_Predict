#!/bin/bash

echo "Submitting jobs"
echo "Running cluster model for 0,1,2,3,4,5,6,7,8,9"

qsub -v VAR1=0 -o /gpfs/project/not81yan/Model/cluster_model/logs/HPC_c_mod2_0.log cluster_model/run.sh
qsub -v VAR1=1 -o /gpfs/project/not81yan/Model/cluster_model/logs/HPC_c_mod2_1.log cluster_model/run.sh
qsub -v VAR1=2 -o /gpfs/project/not81yan/Model/cluster_model/logs/HPC_c_mod2_2.log cluster_model/run.sh
qsub -v VAR1=3 -o /gpfs/project/not81yan/Model/cluster_model/logs/HPC_c_mod2_3.log cluster_model/run.sh
qsub -v VAR1=4 -o /gpfs/project/not81yan/Model/cluster_model/logs/HPC_c_mod2_4.log cluster_model/run.sh
qsub -v VAR1=5 -o /gpfs/project/not81yan/Model/cluster_model/logs/HPC_c_mod2_5.log cluster_model/run.sh
qsub -v VAR1=6 -o /gpfs/project/not81yan/Model/cluster_model/logs/HPC_c_mod2_6.log cluster_model/run.sh
qsub -v VAR1=7 -o /gpfs/project/not81yan/Model/cluster_model/logs/HPC_c_mod2_7.log cluster_model/run.sh
qsub -v VAR1=8 -o /gpfs/project/not81yan/Model/cluster_model/logs/HPC_c_mod2_8.log cluster_model/run.sh
qsub -v VAR1=9 -o /gpfs/project/not81yan/Model/cluster_model/logs/HPC_c_mod2_9.log cluster_model/run.sh
