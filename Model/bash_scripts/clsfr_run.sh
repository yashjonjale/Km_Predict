#!/bin/bash

echo "Submitting jobs"
echo "Running classifier with evals - 200 and sz - 0.4,0.5,0.6,0.7,0.8"


qsub -v VAR1=0.4 -o /home/not81yan/km_predict_proj/Model/logs/HPC_clsfr_train_jul_1_200_0.4.log ./bash_scripts/clsfr.sh
qsub -v VAR1=0.5 -o /home/not81yan/km_predict_proj/Model/logs/HPC_clsfr_train_jul_1_200_0.5.log ./bash_scripts/clsfr.sh
qsub -v VAR1=0.6 -o /home/not81yan/km_predict_proj/Model/logs/HPC_clsfr_train_jul_1_200_0.6.log ./bash_scripts/clsfr.sh
qsub -v VAR1=0.7 -o /home/not81yan/km_predict_proj/Model/logs/HPC_clsfr_train_jul_1_200_0.7.log ./bash_scripts/clsfr.sh
qsub -v VAR1=0.8 -o /home/not81yan/km_predict_proj/Model/logs/HPC_clsfr_train_jul_1_200_0.8.log ./bash_scripts/clsfr.sh


# echo "Running classifier with evals- 5,10,20,40,50,100,150,200,250 and sz - 0.8"

# qsub -v VAR1=5 -o /home/not81yan/km_predict_proj/Model/logs/HPC_clsfr_train_jul_1_5_0.8.log ./bash_scripts/clsfr.sh
# qsub -v VAR1=10 -o /home/not81yan/km_predict_proj/Model/logs/HPC_clsfr_train_jul_1_10_0.8.log ./bash_scripts/clsfr.sh
# qsub -v VAR1=20 -o /home/not81yan/km_predict_proj/Model/logs/HPC_clsfr_train_jul_1_20_0.8.log ./bash_scripts/clsfr.sh
# qsub -v VAR1=40 -o /home/not81yan/km_predict_proj/Model/logs/HPC_clsfr_train_jul_1_40_0.8.log ./bash_scripts/clsfr.sh
# qsub -v VAR1=50 -o /home/not81yan/km_predict_proj/Model/logs/HPC_clsfr_train_jul_1_50_0.8.log ./bash_scripts/clsfr.sh
# qsub -v VAR1=100 -o /home/not81yan/km_predict_proj/Model/logs/HPC_clsfr_train_jul_1_100_0.8.log ./bash_scripts/clsfr.sh
# qsub -v VAR1=150 -o /home/not81yan/km_predict_proj/Model/logs/HPC_clsfr_train_jul_1_150_0.8.log ./bash_scripts/clsfr.sh
# qsub -v VAR1=200 -o /home/not81yan/km_predict_proj/Model/logs/HPC_clsfr_train_jul_1_200_0.8.log ./bash_scripts/clsfr.sh
# qsub -v VAR1=250 -o /home/not81yan/km_predict_proj/Model/logs/HPC_clsfr_train_jul_1_250_0.8.log ./bash_scripts/clsfr.sh
