#!/bin/bash
#PBS -l walltime=10:00:00
#PBS -l select=1:ncpus=1:mem=10GB:ngpus=1:accelerator_model=teslat4
#PBS -A km_enz_var
#PBS -N Align
#PBS -j oe
#PBS -o /home/not81yan/km_predict_proj/Dataset/logs/HPC_Align_jun_26_1.log
#PBS -r y
#PBS -m ae
#PBS -M not81yan@hhu.de

# source $HOME/.bashrc
# source $HOME/.bash_profile
cd /home/not81yan/km_predict_proj/Dataset/
# pwd
module load Python/3.6.5
pip install --user -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de typing-extensions==3.7.4 --upgrade numpy

pip install --user -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de typing-extensions==3.7.4 --upgrade pandas
pip install --user -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de typing-extensions==3.7.4 --upgrade biopython
pip install --user -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de typing-extensions==3.7.4 --upgrade torch
pip install --user -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de typing-extensions==3.7.4 --upgrade fair-esm


LOGFILE="./logs/pipeline_III_$(date '+%Y-%m-%d %H:%M:%S').log"

# Function to log messages with timestamps
log_message() {
    local MESSAGE="$1"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $MESSAGE" >> "$LOGFILE"
}

echo "All modules installed"
log_message "All modules installed"


# python3 ./bash_scripts_subm/check.py



# if [ $? -eq 1 ]; then
#     echo "CUDA ERROR"
#     log_message "CUDA ERROR"
#     exit 1
# fi

# Continue with the rest of the bash script if the exit code was not 1
# echo "Python script completed successfully, continuing bash script execution."
# log_message "Python script completed successfully, continuing bash script execution."


echo "Starting Align"
log_message "Starting Align"


python3 ./final_pipeline/add_align_scores.py


echo "Dataset Align completed successfully"
log_message "Dataset Align completed successfully"

