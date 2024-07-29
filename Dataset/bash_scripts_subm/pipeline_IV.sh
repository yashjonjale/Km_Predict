#!/bin/bash
#PBS -l walltime=10:00:00
#PBS -l select=1:ncpus=2:mem=50GB:ngpus=1:accelerator_model=gtx1080ti
#PBS -A km_enz_var
#PBS -N pipeline_IV
#PBS -j oe
#PBS -o /home/not81yan/km_predict_proj/Dataset/logs/HPC_pipeline_IV_jun_14_1.log
#PBS -r y
#PBS -m ae
#PBS -M not81yan@hhu.de

source $HOME/.bashrc
source $HOME/.bash_profile
cd /home/not81yan/km_predict_proj/Dataset/
module load Python/3.6.5
pip install --user -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de typing-extensions==3.7.4 --upgrade numpy

pip install --user -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de typing-extensions==3.7.4 --upgrade pandas
pip install --user -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de typing-extensions==3.7.4 --upgrade biopython
pip install --user -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de typing-extensions==3.7.4 --upgrade torch
pip install --user -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de typing-extensions==3.7.4 --upgrade fair-esm


LOGFILE="./logs/pipeline_IV_jun_14_1.log"

# Function to log messages with timestamps
log_message() {
    local MESSAGE="$1"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $MESSAGE" >> "$LOGFILE"
}

echo "All modules installed"
log_message "All modules installed"


python3 ./bash_scripts_subm/check.py



if [ $? -eq 1 ]; then
    echo "CUDA ERROR"
    log_message "CUDA ERROR"
    exit 1
fi

# Continue with the rest of the bash script if the exit code was not 1
echo "Python script completed successfully, continuing bash script execution."
log_message "Python script completed successfully, continuing bash script execution."


echo "Starting Pipeline IV"
log_message "Starting Pipeline IV"

echo "Running into_numpy.py"
log_message "Running into_numpy.py"
python3 ./final_pipeline/into_numpy.py

echo "Done with into_numpy.py"
log_message "Done with into_numpy.py"



echo "Dataset Pipeline - IV completed successfully"
log_message "Dataset Pipeline - IV completed successfully"

