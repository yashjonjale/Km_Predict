#!/bin/bash
#PBS -l walltime=10:00:00
#PBS -l select=1:ncpus=4:mem=20GB:ngpus=1:accelerator_model=rtx8000
#PBS -A km_enz_var
#PBS -N Cluster
#PBS -j oe
#PBS -o /gpfs/project/not81yan/Dataset/cluster/logs/HPC_cluster.log
#PBS -r y
#PBS -m ae
#PBS -M not81yan@hhu.de

source $HOME/.bashrc
source $HOME/.bash_profile
cd /gpfs/project/not81yan/Dataset/
module load Python/3.6.5
pip install --user -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de typing-extensions==3.7.4 --upgrade numpy

pip install --user -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de typing-extensions==3.7.4 --upgrade pandas
pip install --user -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de typing-extensions==3.7.4 --upgrade biopython
pip install --user -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de typing-extensions==3.7.4 --upgrade torch
pip install --user -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de typing-extensions==3.7.4 --upgrade fair-esm



LOGFILE="./cluster/logs/cluster_script.log"

# Function to log messages with timestamps
log_message() {
    local MESSAGE="$1"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $MESSAGE" >> "$LOGFILE"
}

echo "All modules installed"
log_message "All modules installed"


python3 ./bash_scripts_subm/check.py > check



if [ $? -eq 1 ]; then
    echo "CUDA ERROR"
    log_message "CUDA ERROR"
    exit 1
fi

# Continue with the rest of the bash script if the exit code was not 1
echo "Python script completed successfully, continuing bash script execution."
log_message "Python script completed successfully, continuing bash script execution."


echo "Starting ESM addition"
log_message "Starting ESM addition"

echo "Running add_esm.py"
log_message "Running add_esm.py"
python3 ./cluster/cluster.py

echo "Done with add_esm.py"
log_message "Done with add_esm.py"



echo "Dataset ESM addition completed successfully"
log_message "Dataset ESM addition completed successfully"

