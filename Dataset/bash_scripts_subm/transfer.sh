#!/bin/bash

mkdir sub_files_transfer
scp -r not81yan@storage.hpc.rz.uni-duesseldorf.de:/home/not81yan/km_predict_proj/Dataset/sub_files_transfer /home/yashjonjale/Documents/Dataset
echo "Transfer completed successfully"
sleep 6
#Now run add_seq.py locally

# mkdir refined_data_seq

echo "Running add_seq.py"
python3 ./final_pipeline/add_seq.py
echo "Done with add_seq.py"

scp -r ./refined_data_seq/ not81yan@storage.hpc.rz.uni-duesseldorf.de:/home/not81yan/km_predict_proj/Dataset/
echo "Local files transferred successfully"