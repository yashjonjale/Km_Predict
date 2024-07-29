import torch
import esm
import os
import pandas as pd
import numpy as np
import glob
import json
# from Bio import pairwise2
# from Bio.pairwise2 import format_alignment
import time
from Bio import Align
emb_dim = 1280
import sys
import logging
import logging.handlers

def setup_logger(log_file_path):
    # Create a logger
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)  # Set the logging level

    # Create a file handler to log messages to a file
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)  # Set the logging level for the file handler

    # Create a console handler to log messages to the console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # Set the logging level for the console handler

    # Create a formatter and set it for both handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger



def extract_embeddings(data, model, device, batch_converter):
    # Prepare the input
    # batch_converter = model.alphabet.get_batch_converter()
    # data = lst
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    # Move the input data to the GPU if available
    batch_tokens = batch_tokens.to(device)

    # Encode the sequences
    with torch.no_grad():
        results = model(batch_tokens.to(device), repr_layers=[33], return_contacts=False)

    # Extract the embeddings
    token_representations = results["representations"][33].cpu().numpy()
    # print(token_embeddings.shape)  # This should give you (num_sequences, sequence_length, embedding_dim)
    # return token_embeddings
    protein_representations = np.mean(token_representations[:, 1:-1, :], axis=1)
    return protein_representations ## num_sequences, embedding_dim

def similarity_score_vecs(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
def similarity_seqs(seq1, seq2):
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    scores = [alignment.score for alignment in aligner.align(seq1, seq2)]
    return max(scores) / max(len(seq1), len(seq2))



# start the logger
log_file_path = './logs/add_esm_vecs.log'
logger = setup_logger(log_file_path)
logger.info("Starting add_esm_vecs.py")





## get current working directory
cwd = os.getcwd()
## convert to string
cwd = str(cwd)
logger.info(f"Current working directory: {cwd}")
logger.info("Loading the ESM model")
model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_location = cwd+"/model/esm1b_t33_650M_UR50S.pt")
batch_converter = alphabet.get_batch_converter()
device = torch.device("cuda")
model = model.to(device)  # Move the model to the GPU if available
model.eval()

logger.info("Model loaded successfully")



## extract the substr2substr_id

# with open(os.getcwd()+'/cache_dir/substr2substr_id.json', 'r') as f:
    # substr2substr_id = json.load(f)

# with open(os.getcwd()+'/cache_dir/substr_id2substr.json', 'r') as f:
    # substr_id2substr = json.load(f)
# logger.info("Substr2substr_id and substr_id2substr loaded successfully")

# fnames = []

directory = os.getcwd()+'/refined_data_seq/'
naming_format = '*sub_seq.csv'  # This will match any file that ends with .json

# uni2seq = {}
files = glob.glob(f'{directory}/{naming_format}')
print(f"Total files: {len(files)}")
# input("Press Enter to continue...")
logger.info(f"Total files: {len(files)}")
start = time.time()

i=0

for file in files:
    ec_num = file.strip().split('/')[-1].split('_')[0]
    sub_id = file.strip().split('_')[-3]
    # print(f"File: {file}")
    # print(f"EC: {ec_num}, Sub_id: {sub_id}")
    # input("Press Enter to continue...")
    ## open as dataframe
    df = pd.read_csv(file)
    ## now add more columns to the dataframe for the esm vectors with default value '-'
    # df['seq_str'] = '-'
    df['esm'] = '-'
    # df['esm_wild'] = '-'
    # df['sim_seqs'] = '-'
    # df['sim_vecs'] = '-'
    len_cor = True
    #iterate over the rows
    for index, row in df.iterrows():
        
        seq = row['seq_str']
        if seq is '-':
            df.drop(index, inplace=True)
            continue
        if seq == 'ignore_entry':
            df.drop(index, inplace=True)
            continue
        if len(seq) > 1000:
            df.drop(index, inplace=True)
            continue
        data = [
            ("label1", seq)
        ]
        vecs = extract_embeddings(data, model, device, batch_converter)
        df.at[index, 'esm'] = list(vecs[0])
    # seq_sim tells you if mutated or not
    ## save the dataframe   
    # print(f"Saving {ec_num}_{sub_id}_sub_esm.csv")
    if len_cor:
        df.to_csv(directory+"../refined_data_esm/"+f"{ec_num}_{sub_id}_sub_seq_esm.csv")
    i+=1
    if i%10 == 0:
        logger.info(f"Done {i} files, time elapsed: {time.time()-start}")
        print(f"Done {i} files, time elapsed: ", time.time()-start)            
        
print("Done, time elapsed: ", time.time()-start)
logger.info(f"Done, time elapsed: {time.time()-start}")

    
