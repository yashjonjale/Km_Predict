#!/bin/bash

echo "Submitting jobs"
echo "Running model 5 for sizes 5, 10, 20, 30, 50, 100, 150, 200, 250, 300"

qsub -v VAR1=5 -o /home/not81yan/km_predict_proj/Model/logs/HPC_model_5_train_jun_26_1_5.log ./bash_scripts/model_5_test.sh
qsub -v VAR1=10 -o /home/not81yan/km_predict_proj/Model/logs/HPC_model_5_train_jun_26_1_10.log ./bash_scripts/model_5_test.sh
qsub -v VAR1=20 -o /home/not81yan/km_predict_proj/Model/logs/HPC_model_5_train_jun_26_1_20.log ./bash_scripts/model_5_test.sh
qsub -v VAR1=30 -o /home/not81yan/km_predict_proj/Model/logs/HPC_model_5_train_jun_26_1_30.log ./bash_scripts/model_5_test.sh
qsub -v VAR1=50 -o /home/not81yan/km_predict_proj/Model/logs/HPC_model_5_train_jun_26_1_50.log ./bash_scripts/model_5_test.sh
qsub -v VAR1=100 -o /home/not81yan/km_predict_proj/Model/logs/HPC_model_5_train_jun_26_1_100.log ./bash_scripts/model_5_test.sh
qsub -v VAR1=150 -o /home/not81yan/km_predict_proj/Model/logs/HPC_model_5_train_jun_26_1_150.log ./bash_scripts/model_5_test.sh
qsub -v VAR1=200 -o /home/not81yan/km_predict_proj/Model/logs/HPC_model_5_train_jun_26_1_0.2.log ./bash_scripts/model_5_test.sh
qsub -v VAR1=250 -o /home/not81yan/km_predict_proj/Model/logs/HPC_model_5_train_jun_26_1_250.log ./bash_scripts/model_5_test.sh
qsub -v VAR1=300 -o /home/not81yan/km_predict_proj/Model/logs/HPC_model_5_train_jun_26_1_300.log ./bash_scripts/model_5_test.sh
