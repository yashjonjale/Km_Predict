import pandas as pd
import numpy as np
import os
import glob
import ast
import time
from Bio import Align
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
log_file_path = os.getcwd() + '/logs/add_align_scores.log'
logger = setup_logger(log_file_path)
logger.info('Starting add_align_scores.py')





def edit_distance(str1, str2):
    # Get the length of both strings
    len_str1 = len(str1)
    len_str2 = len(str2)

    # Create a 2D array to store results of subproblems
    dp = [[0 for _ in range(len_str2 + 1)] for _ in range(len_str1 + 1)]

    # Initialize the dp array
    for i in range(len_str1 + 1):
        for j in range(len_str2 + 1):

            # If the first string is empty, only option is to insert all characters of the second string
            if i == 0:
                dp[i][j] = j    # Min. operations = j

            # If the second string is empty, only option is to remove all characters of the first string
            elif j == 0:
                dp[i][j] = i    # Min. operations = i

            # If the last characters are the same, ignore the last character and recur for the remaining substring
            elif str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]

            # If the last characters are different, consider all possibilities and find the minimum
            else:
                dp[i][j] = 1 + min(dp[i][j-1],    # Insert
                                   dp[i-1][j],    # Remove
                                   dp[i-1][j-1])  # Replace

    # The answer is in the cell dp[len_str1][len_str2]
    return dp[len_str1][len_str2]


def similarity_seqs(seq1, seq2):
#     aligner = Align.PairwiseAligner()
#     aligner.mode = 'global'
#     scores = [alignment.score for alignment in aligner.align(seq1, seq2)]
    return 1-(edit_distance(seq1,seq2) / min(len(seq1), len(seq2)))


def similarity_score_vecs(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


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
    # print(df.head())
    for index, row in df.iterrows():
        sim_vec = row['sim_vecs']
        sim_seq = similarity_seqs(row['seq_w'], row['seq_m'])
        sim_s.append(sim_seq)
        sim_v.append(sim_vec)
        delta_km.append(row['value_diff'])
        # update the row
        df.at[index, 'sim_seqs'] = sim_seq
    # save the updated dataframe
    # print(df.head())
    df.to_csv(file)
    j += 1
    # input("Press Enter to continue...")
    if j % 10 == 0:
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
