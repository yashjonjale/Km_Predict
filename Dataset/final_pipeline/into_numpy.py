import pandas as pd
import numpy as np
import os
import glob
import ast
import time

path = os.getcwd() + '/paired_data/'
naming_format = '*pairs.csv'
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
log_file_path = os.getcwd() + '/final_dat/final_data.log'
logger = setup_logger(log_file_path)
logger.info('Starting into_numpy.py')


def similarity_score_vecs(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

i = 0
start = time.time()



# print number of files
print(f"Number of files: {len(files)}")
logger.info(f"Number of files: {len(files)}")

collection  = []
entries = 0
for file in files:
    # print(f"Processing {file}")
    ec_num = file.strip().split('/')[-1].split('_')[0] 
    sub_id = file.strip().split('_')[-2]
    # print(f"EC number: {ec_num}")
    # print(f"Substrate ID: {sub_id}")
    df = pd.read_csv(file)
    entries += len(df)
    # print(f"Processing {file}")
    # logger.info(f"Processing {file}")
    # print(df.head())
    #delete columns - 'sim_seqs' and 'sim_vecs'
    # print(df.columns)
    # print(df.head())
    # print(df.shape)
    # print(df.dtypes)
    # print(df.describe())
    df['sim_vecs'] = df['sim_vecs'].apply(lambda x: float(x))
    sim_vecs = df['sim_vecs'].values
    df.drop(['sim_vecs'], axis=1, inplace=True)
    
    df['esm_wild'] = df['esm_wild'].apply(ast.literal_eval)
    df['esm_diff'] = df['esm_diff'].apply(ast.literal_eval)
    df['km_wild'] = df['km_wild'].apply(lambda x: float(x))
    df['value_diff'] = df['value_diff'].apply(lambda x: float(x))
    
    target_values = df['value_diff'].values
    km_wild = df['km_wild'].values
    diff = np.stack(df['esm_diff'].values)
    wild = np.stack(df['esm_wild'].values)
    result = np.column_stack((sim_vecs,km_wild,target_values, wild, diff))
    collection.append(result)
    i += 1
    if i % 10 == 0:
        print(f"Processed {i} files and entries {entries}, time elapsed: {time.time()-start}")
        logger.info(f"Processed {i} files and entries {entries}, time elapsed: {time.time()-start}")
    
final_data = np.vstack(collection)


#save the final data in compressed format
np.savez_compressed('./final_dat/final_data.npz', final_data)

print(f"Time taken: {time.time()-start}")
logger.info(f"Time taken: {time.time()-start}")
print(f"Total files: {len(files)}")
logger.info(f"Total files: {len(files)}")
print(f"Total pairs: {entries}")
logger.info(f"Total pairs: {entries}")

print(f"Final data shape: {final_data.shape}")