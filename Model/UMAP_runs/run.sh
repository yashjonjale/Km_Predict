#!/bin/bash
#PBS -l walltime=38:00:00
#PBS -l select=1:ncpus=1:mem=30GB:ngpus=1:accelerator_model=gtx1080ti
#PBS -A km_enz_var
#PBS -N classifier
#PBS -j oe
## PBS -o /gpfs/project/not81yan/Model/UMAP_runs/logs/umap_jul_11.log
#PBS -r y
#PBS -m ae
#PBS -M not81yan@hhu.de




source $HOME/.bashrc
source $HOME/.bash_profile
cd /gpfs/project/not81yan/Model/
pwd
module load Python/3.6.5
module load Miniconda/3
conda activate yash

# pip install --user -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de typing-extensions==3.7.4 --upgrade numpy
# pip install --user -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de typing-extensions==3.7.4 --upgrade pandas
# pip install --user -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de typing-extensions==3.7.4 --upgrade sklearn
# pip install --user -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de typing-extensions==3.7.4 --upgrade scipy
# pip install --user -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de typing-extensions==3.7.4 --upgrade optuna




LOGFILE="./UMAP_runs/logs/umap_jul_11.log"

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


echo "Starting the model training"
log message "Starting the model training"


python3 ./UMAP_runs/umap_run.py --components 50


echo "Done with model.py"
log_message "Done with model.py"


echo "completed successfully"
log_message "completed successfully"

