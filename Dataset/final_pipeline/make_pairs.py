import pandas as pd
import numpy as np
import os
import glob
import ast
import time
from Bio import Align

mean_type = 'num_value_am'
path = os.getcwd() + '/refined_data_esm/'
naming_format = '*sub_seq_esm.csv'
files = glob.glob(f'{path}/{naming_format}')
THRESHOLD = 0.9


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


#start logger   
log_file_path = os.getcwd() + '/logs/make_pairs_sim_seq.log'
logger = setup_logger(log_file_path)
logger.info('Starting make_pairs.py')




def similarity_seqs(seq1, seq2):
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    scores = [alignment.score for alignment in aligner.align(seq1, seq2)]
    return max(scores) / max(len(seq1), len(seq2))






def similarity_score_vecs(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

i = 0
start = time.time()
sim_s = []
sim_v = []
delta_km = []


# print number of files
print(f"Number of files: {len(files)}")
logger.info(f"Number of files: {len(files)}")
num_pairs = 0
saved = 0
for file in files:
    # print(f"Processing file: {file}")
    pairs = pd.DataFrame(columns=['km_wild','value_diff', 'esm_wild', 'esm_diff', 'seq_w','seq_m','sim_vecs','sim_seqs'])
    ec_num = file.strip().split('/')[-1].split('_')[0] 
    sub_id = file.strip().split('_')[-4]
    df = pd.read_csv(file)
    # print(df.head())
    df['esm'] = df['esm'].apply(ast.literal_eval)
    df['num_value_gm'] = df['num_value_gm'].apply(lambda x: float(x))
    df['num_value_am'] = df['num_value_am'].apply(lambda x: float(x))
    
    if len(df) < 2:
        continue
    saved += 1
    for index, row in df.iterrows():
        for index2, row2 in df.iterrows():
            if index == index2:
                continue
            # print the rows
            # print(row)
            # print(row2)
            t1 = row['sim_seqs']
            t2 = row2['sim_seqs']
            esm_w = np.array(row['esm'])
            esm_m = np.array(row2['esm'])
            diff = esm_m - esm_w
            seq1 = row['seq_str']
            seq2 = row2['seq_str']
            # sim_seq = similarity_seqs(seq1, seq2)
            sim_vec = similarity_score_vecs(esm_w, esm_m)
            # sim_s.append(sim_seq)
            # sim_v.append(sim_vec)
            del_km = np.log10(row2['num_value_gm']) - np.log10(row['num_value_gm'])
            # add a row to the pairs dataframe
            # delta_km.append(del_km)
            pairs = pairs._append({ 'km_wild':np.log10(row['num_value_gm']),'value_diff': del_km, 'esm_wild': list(esm_w), 'esm_diff': list(diff), 'seq_w':seq1, 'seq_m':seq2, 'sim_vecs': sim_vec, 'sim_seqs': 0}, ignore_index=True)
            # print(pairs.head())
    num_pairs += len(pairs)
    # print(pairs.head())
    pairs.to_csv(os.getcwd()+f"/paired_data_sim_seqs/{ec_num}_{sub_id}_pairs.csv")
    # input("Press Enter to continue...")
    ## save the pairs dataframe as a csv file
    i += 1
    if i % 10 == 0:
        print(f"Processed {i} files and {num_pairs} pairs, saved {saved} time elapsed: {time.time()-start}")
        logger.info(f"Processed {i} files and {num_pairs} pairs, saved {saved}, time elapsed: {time.time()-start}")
    


# print(f"Time taken: {time.time()-start}")
# logger.info(f"Time taken: {time.time()-start}")
# print(f"Total files: {len(files)} and saved {saved}")
# logger.info(f"Total files: {len(files)} and saved {saved}")
# print(f"Total pairs: {num_pairs}")
# logger.info(f"Total pairs: {num_pairs}")

t1 = time.time()
##start the analysis and sim seqs and sim vecs
path = os.getcwd() + '/paired_data_sim_seqs/'
naming_format = '*_pairs.csv'
files = glob.glob(f'{path}/{naming_format}')
sim_s = []
sim_v = []
delta_km = []
j = 0
for file in files:
    df = pd.read_csv(file)
    print(df.head())
    for index, row in df.iterrows():
        sim_vec = row['sim_vecs']
        sim_seq = similarity_seqs(row['seq_w'], row['seq_m'])
        sim_s.append(sim_seq)
        sim_v.append(sim_vec)
        delta_km.append(row['value_diff'])
        # update the row
        df.at[index, 'sim_seqs'] = sim_seq
    # save the updated dataframe
    print(df.head())
    df.to_csv(file)
    j += 1
    input("Press Enter to continue...")
    if j % 2 == 0:
        print(f"Processed {j} files and time elapsed: {time.time()-t1}")

print(f"Mean similarity score for seqs: {np.mean(sim_s)}")
logger.info(f"Mean similarity score for seqs: {np.mean(sim_s)}")
print(f"Mean similarity score for vecs: {np.mean(sim_v)}")
logger.info(f"Mean similarity score for vecs: {np.mean(sim_v)}")
print(f"Mean delta km: {np.mean(delta_km)}")
logger.info(f"Mean delta km: {np.mean(delta_km)}")

#find the correlation between the similarity scores and the delta km
from scipy.stats import pearsonr
corr, _ = pearsonr(sim_s, delta_km)
print(f"Pearson correlation- sim_s - del_km: {corr}")
logger.info(f"Pearson correlation- sim_s - del_km: {corr}")
corr, _ = pearsonr(sim_v, delta_km)
print(f"Pearson correlation- sim_v - del_km: {corr}")
logger.info(f"Pearson correlation - sim_v - del_km: {corr}")

# also for similarity scores
corr, _ = pearsonr(sim_s, sim_v)

print(f"Pearson correlation sim_s and sim_v: {corr}")
logger.info(f"Pearson correlation sim_s and sim_v: {corr}")

# now find r2 values

from sklearn.metrics import r2_score
r2 = r2_score(sim_v, sim_s)
print(f"R2 score sim_s and sim_v: {r2}")
logger.info(f"R2 score between sim_s and sim_v: {r2}")

#save sim_s, sim_v and delta_km
np.savez(os.getcwd() + '/plots/similarity_scores.npz', sim_s = sim_s, sim_v = sim_v, delta_km = delta_km)

# plot sim_s
import matplotlib.pyplot as plt
plt.hist(sim_s, bins=20)
plt.xlabel('Similarity score seq')
# save it
plt.savefig(os.getcwd() + '/plots/sim_s.png')

# plot sim_v
plt.hist(sim_v, bins=20)
plt.xlabel('Similarity score vec')
# save it
plt.savefig(os.getcwd() + '/plots/sim_v.png')

# make a scatter plot of sim_s and sim_v
plt.scatter(sim_s, sim_v)
plt.xlabel('Similarity score seq')
plt.ylabel('Similarity score vec')
plt.title('Similarity score seq vs vec')
# save it
plt.savefig(os.getcwd() + '/plots/sim_s_vs_sim_v.png')
