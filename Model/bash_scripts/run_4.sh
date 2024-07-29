#!/bin/bash

echo "Submitting jobs"
echo "Running model 4 for sizes 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 0.80"

qsub -v VAR1=0.01 -o /home/not81yan/km_predict_proj/Model/logs/HPC_model_4_train_jun_24_1_0.01.log ./bash_scripts/model_4_test.sh
qsub -v VAR1=0.02 -o /home/not81yan/km_predict_proj/Model/logs/HPC_model_4_train_jun_24_1_0.02.log ./bash_scripts/model_4_test.sh
qsub -v VAR1=0.03 -o /home/not81yan/km_predict_proj/Model/logs/HPC_model_4_train_jun_24_1_0.03.log ./bash_scripts/model_4_test.sh
qsub -v VAR1=0.04 -o /home/not81yan/km_predict_proj/Model/logs/HPC_model_4_train_jun_24_1_0.04.log ./bash_scripts/model_4_test.sh
qsub -v VAR1=0.05 -o /home/not81yan/km_predict_proj/Model/logs/HPC_model_4_train_jun_24_1_0..05.log ./bash_scripts/model_4_test.sh
qsub -v VAR1=0.1 -o /home/not81yan/km_predict_proj/Model/logs/HPC_model_4_train_jun_24_1_0.1.log ./bash_scripts/model_4_test.sh
qsub -v VAR1=0.15 -o /home/not81yan/km_predict_proj/Model/logs/HPC_model_4_train_jun_24_1_0.15.log ./bash_scripts/model_4_test.sh
qsub -v VAR1=0.20 -o /home/not81yan/km_predict_proj/Model/logs/HPC_model_4_train_jun_24_1_0.2.log ./bash_scripts/model_4_test.sh
qsub -v VAR1=0.25 -o /home/not81yan/km_predict_proj/Model/logs/HPC_model_4_train_jun_24_1_0.25.log ./bash_scripts/model_4_test.sh
qsub -v VAR1=0.30 -o /home/not81yan/km_predict_proj/Model/logs/HPC_model_4_train_jun_24_1_0.30.log ./bash_scripts/model_4_test.sh
qsub -v VAR1=0.40 -o /home/not81yan/km_predict_proj/Model/logs/HPC_model_4_train_jun_24_1_0.40.log ./bash_scripts/model_4_test.sh
qsub -v VAR1=0.50 -o /home/not81yan/km_predict_proj/Model/logs/HPC_model_4_train_jun_24_1_0.50.log ./bash_scripts/model_4_test.sh
qsub -v VAR1=0.70 -o /home/not81yan/km_predict_proj/Model/logs/HPC_model_4_train_jun_24_1_0.70.log ./bash_scripts/model_4_test.sh
qsub -v VAR1=0.80 -o /home/not81yan/km_predict_proj/Model/logs/HPC_model_4_train_jun_24_1_0.80.log ./bash_scripts/model_4_test.sh