#!/bin/bash
#PBS -l walltime=10:00:00
#PBS -l select=1:ncpus=1:mem=25GB:ngpus=1:accelerator_model=rtx6000
#PBS -A km_enz_var
#PBS -N Cluster_3
#PBS -j oe
#PBS -o /gpfs/project/not81yan/Model/cluster_model/logs/HPC_cluster_test.log
#PBS -r y
#PBS -m ae
#PBS -M not81yan@hhu.de

#18
#4
#8
#6
#6
#4
#6


source $HOME/.bashrc
source $HOME/.bash_profile
cd /gpfs/project/not81yan/Model/
module load Python/3.6.5
pip install --user -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de typing-extensions==3.7.4 --upgrade numpy
pip install --user -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de typing-extensions==3.7.4 --upgrade matplotlib
pip install --user -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de typing-extensions==3.7.4 --upgrade pandas
pip install --user -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de typing-extensions==3.7.4 --upgrade sklearn
pip install --user -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de typing-extensions==3.7.4 --upgrade scipy
pip install --user -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de typing-extensions==3.7.4 --upgrade hyperopt
pip install --user -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de typing-extensions==3.7.4 --upgrade xgboost

LOGFILE="./cluster_model/logs/cluster_test.log"

# Function to log messages with timestamps
log_message() {
    local MESSAGE="$1"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $MESSAGE" >> "$LOGFILE"
}

echo "All modules installed"
log_message "All modules installed"


python3 ./bash_scripts/check.py



if [ $? -eq 1 ]; then
    echo "CUDA ERROR"
    log_message "CUDA ERROR"
    exit 1
fi

# Continue with the rest of the bash script if the exit code was not 1
echo "Python script completed successfully, continuing bash script execution."
log_message "Python script completed successfully, continuing bash script execution."

# #check if the files - train_pairs_idx.csv and test_pairs_idx.csv are present in the directory, and keep looping until they are present
# while [ ! -f ./refined_model/data/train_pairs_idx.csv ] || [ ! -f ./refined_model/data/test_pairs_idx.csv ]
# do
#     # echo "Waiting for the files - train_pairs_idx.csv and test_pairs_idx.csv to be present in the directory"
#     log_message "Waiting for the files - train_pairs_idx.csv and test_pairs_idx.csv to be present in the directory"
#     sleep 600
# done


echo "Starting filtered model"
log_message "Starting filtered model"

echo "Running model_fil.py"
log_message "Running model_fil.py"
python3 ./cluster_model/c_mod_3.py --index 4 

echo "Done with model_fil.py"
log_message "Done with model_fil.py"



echo "Dataset filtered model completed successfully"
log_message "Dataset filtered model completed successfully"

