import pandas
import json
import time
import glob
from Bio import SwissProt
from Bio import ExPASy
from scipy.stats import gmean
import os

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



def gm(itr1):
    prod = 1
    i = 0
    for x in itr1:
        prod*=x
        i+=1
    return prod**(1/i)

def geometric_mean(series):
    return gmean(series)


def extract_sequence(uniprot_id):
    handle = ExPASy.get_sprot_raw(uniprot_id)
    record = SwissProt.read(handle)
    return record.sequence
def safe_extract_sequence(uniprot_id):
    try:
        return extract_sequence(uniprot_id)
    except ValueError:
        return None  # or '', or any default value you want to use

#start the logger
log_file_path = './logs/pre_process_tables.log'
logger = setup_logger(log_file_path)
logger.info("Starting pre_process_tables.py")


cwd = os.getcwd()
directory = cwd+"/cache_dir/"
naming_format = '*-km.csv'  # This will match any file that ends with .json

km_csv_files = glob.glob(f'{directory}/{naming_format}')

substr2substr_id = {}
substr_id2substr = {}
idnum=0

total_entries = 0
num_entries = 0

logger.info(f"Total files to be processed (one EC class per file): {len(km_csv_files)}")

for file_name in km_csv_files:
    km_df = pandas.read_csv(file_name)
    print(f"Processing {file_name}")
    print(km_df.head())
    # input("Press Enter to continue...")
    if 'proteins' not in km_df.columns:
        # print("No proteins column found in the file")
        continue
    ## drop the comments column
    if 'comment' in km_df.columns:
        km_df.drop(columns=['comment'], inplace=True)
    if 'organisms' in km_df.columns:
        km_df.drop(columns=['organisms'], inplace=True)
    if 'references' in km_df.columns:
        km_df.drop(columns=['references'], inplace=True)
    if 'Unnamed: 0' in km_df.columns:
        km_df.drop(columns=['Unnamed: 0'], inplace=True)
    ## make a new column in km_df called "seq_orig" and "seq_final" for all rows
    for index, row in km_df.iterrows():
        if row['uniprot'] == "ignore_entry":
            ## delete this row
            km_df.drop(index, inplace=True)
            continue
        # seq = extract_sequence(row['uniprot'])
        # km_df.at[index, 'seq_orig'] = seq
    # print(km_df.head())
    # print(km_df.columns)
    ## check if df is empty
    if km_df.empty:
        continue
    print(km_df.head())
    grouped = km_df.groupby(['proteins', 'value', 'change','mutation','uniprot']).agg(
        num_value_gm = ('num_value', geometric_mean),
        num_value_am = ('num_value', 'mean'),
    ).reset_index()
    print(grouped.head())
    # print(grouped.columns)
    ## iterate over the grouped dataframe
    ## create a new column called "uniseq" in the grouped dataframe
    ## for each row in the grouped dataframe, extract the sequence from uniprot
    ## and store it in the "uniseq" column
    ## drop the rows where either num_value_gm or num_value_am is zero

    # grouped['uniseq'] = [safe_extract_sequence(x) for x in grouped['uniprot']]
    # grouped = grouped[grouped['uniseq'].notna()]  # drop rows where 'uniseq' is None
    grouped1 = grouped.groupby('value')
    dfs = {group: data for group, data in grouped1}
    f_fname = ".".join(file_name.strip().split('/')[-1].split('.')[0:-1])
    
    
    
    
    for key, df_group in dfs.items():
        print(f"Group: {key}")
        if key not in substr2substr_id.keys():
            substr2substr_id[key] = idnum
            substr_id2substr[idnum] = key
            idnum+=1
        ##check if there is an entry for which the mutation column is 'mutated'
        ## if there is not then continue
        # if 'mutated' not in df_group['mutation'].values:
        #     continue
        df_group.drop(columns=['value'], inplace=True)
        df_group.drop(columns=['proteins'], inplace=True)
        # print(df_group)
        ## save the df_group to a new csv file
        total_entries+= df_group.shape[0]
        num_entries+=1
        # print(f"Saving to {directory}{f_fname}_{str(substr2substr_id[key])}_sub.csv")
        df_group.to_csv(directory+f_fname+"_"+str(substr2substr_id[key])+"_sub"+".csv", index=False) 
        # input("Press Enter to continue...")
    # print(f"Saving to {directory}{f_fname}_new.csv")
    # grouped.to_csv(directory+".".join(file_name.strip().split('/')[-1].split('.')[0:-1])+"_new.csv", index=False)
    # input("Press Enter to continue...")
logger.info(f"dumping both substrate id mappings to json files")
with open(directory+"substr2substr_id.json", "w") as f:
    json.dump(substr2substr_id, f)

with open(directory+"substr_id2substr.json", "w") as f:
    json.dump(substr_id2substr, f)
logger.info("Dumping over")
print(f"Total entries: {total_entries}")
print(f"Total number of ec/substrates: {num_entries}")
print(f"Total number of unique substrates: {len(substr2substr_id)}")
print(f"Mean of entries in a table {total_entries/num_entries}")
logger.info(f"Total entries: {total_entries}")
logger.info(f"Total number of ec/substrates: {num_entries}")
logger.info(f"Total number of unique substrates: {len(substr2substr_id)}")
logger.info(f"Mean of entries in a table {total_entries/num_entries}")
logger.info("Done with all EC numbers")

logger.info("Pre-processing done")
