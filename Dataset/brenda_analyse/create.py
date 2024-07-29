import json
from Bio import ExPASy
from Bio import SwissProt
import pandas


import sys
import logging
import logging.handlers
import os

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


# start the logger
log_file_path = './logs/create.log'
logger = setup_logger(log_file_path)
logger.info("Starting create.py")

logger.info("Opening the brenda file")
brenda_path = os.getcwd()+'/cache_dir/brenda_2023_1.json'

def open_as_dict(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

Brenda_dict = open_as_dict(brenda_path)
logger.info("Brenda file opened successfully")

## important stuff -> km_values, engg, references, organism, 
 
def get_imp_stuff(Brenda_dict, ec_num):
    if ec_num not in Brenda_dict['data']:
        return None, None, None, None, None
    if Brenda_dict['data'][ec_num] is []:
        return None, None, None, None, None
    if 'km_value' not in Brenda_dict['data'][ec_num].keys():
        return None, None, None, None, None
    if 'proteins' not in Brenda_dict['data'][ec_num].keys():
        return None, None, None, None, None
    if 'engineering' not in Brenda_dict['data'][ec_num].keys():
        return None, None, None, None, None
    # if 'commment' not in Brenda_dict['data'][ec_num]['engineering'][0]:
        # print(Brenda_dict['data'][ec_num])
        # return None, None, None, None, None
    km_vals = Brenda_dict['data'][ec_num]['km_value']
    engg = Brenda_dict['data'][ec_num]['engineering']
    references = Brenda_dict['data'][ec_num]['references']
    organisms = Brenda_dict['data'][ec_num]['organisms']
    proteins = Brenda_dict['data'][ec_num]['proteins']
    return km_vals, engg, references, organisms, proteins

def extract_uniprot_accessions(id, proteins):
    if id not in proteins.keys():
        return "ignore_entry"
    if 'source' in proteins[id][0] and proteins[id][0]['source'] == 'uniprot':
        return proteins[id][0]['accessions'][0]
    else:
        return "ignore_entry" ##implies ignore
    
def fix_km_val_refs(km_vals):
    new_km_vals = []
    for km_val in km_vals:
        if 'comment' not in km_val.keys():
            refnum = km_val['references'][0]
            if 'proteins' not in km_val.keys():
                continue
            pronum = km_val['proteins'][0]
            n_km_val = km_val.copy()
            n_km_val['comment'] = '-'
            n_km_val['mutation'] = '-'
            n_km_val['references'] = refnum
            
            n_km_val['proteins'] = pronum

            new_km_vals.append(n_km_val)
            continue
        comment = km_val['comment']
        comments = comment.strip().split(';')
        for c in comments:
            tokens = c.strip().split(' ')
            pronum = tokens[0][1:-1]
            refnum = tokens[-1][1:-1]
            n_km_val = km_val.copy()
            n_km_val['comment'] = c
            n_km_val['mutation'] = '-'
            if 'mutant' in tokens:
                n_km_val['mutation'] = 'mutated'
            if 'recombinant' in tokens:
                n_km_val['mutation'] = 'recombinant'
            n_km_val['references'] = refnum
            n_km_val['proteins'] = pronum
            new_km_vals.append(n_km_val)
    return new_km_vals

def fix_engg_engg_refs(engg):
    new_engg = []
    for engg_val in engg:
        if 'proteins' not in engg_val.keys():
            continue
        if 'comment' not in engg_val.keys():
            # print(engg_val)
            # input("______")
            refnum = engg_val['references'][0]
            
            pronum = engg_val['proteins'][0]
            n_engg_val = engg_val.copy()
            n_engg_val['comment'] = '-'
            n_engg_val['references'] = refnum
            n_engg_val['proteins'] = pronum
            new_engg.append(n_engg_val)
            continue
        comment = engg_val['comment']
        comments = comment.strip().split(';')
        for c in comments:
            tokens = c.strip().split(' ')
            pronum = tokens[0][1:-1]
            refnum = tokens[-1][1:-1]
            n_engg_val = engg_val.copy()
            n_engg_val['comment'] = c
            n_engg_val['references'] = refnum
            n_engg_val['proteins'] = pronum
            new_engg.append(n_engg_val)
    return new_engg

## Loop over the ec numbers and extract the important stuff

ec_nums = list(Brenda_dict['data'].keys())
data = []

## start measuring time
import time
start = time.time()
i=0
for ec_num in ec_nums:
    i+=1
    if i%100 == 0:
        print(f"Done with {i} EC numbers and time elapsed is {time.time()-start} seconds")
        logger.info(f"Done with {i} EC numbers and time elapsed is {time.time()-start} seconds")
    
    # print("EC Number: ", ec_num)
    if len(Brenda_dict['data'][ec_num]) == 0:
        continue
    km_vals, engg, references, organisms, proteins = get_imp_stuff(Brenda_dict, ec_num)
    if engg is None:
        continue
    if proteins is None:
        continue
    km_vals = fix_km_val_refs(km_vals)
    engg = fix_engg_engg_refs(engg)
    for dict_ in km_vals:
        if 'num_value' not in dict_:
            dict_['num_value'] = '-'  # or any default value you want to use
        if 'min_value' not in dict_:
            dict_['min_value'] = '-'
        if 'max_value' not in dict_:
            dict_['max_value'] = '-'
    for dict_ in engg:
        if 'organisms' not in dict_:
            dict_['organisms'] = '-'
    ## create dataframes for km_vals and engg
    km_df = pandas.DataFrame(km_vals)
    engg_df = pandas.DataFrame(engg)
    ## drop the column organisms in engg_df
    if 'organisms' in engg_df.columns:
        engg_df = engg_df.drop(columns=['organisms'])
    ## Now iterate over the rows of km_df, and if num_value is None, then assign the geometric mean of min_value and max_value to num_value    
    
    for index, row in km_df.iterrows():
        if row['num_value'] is '-':
            min_val = row['min_value']
            max_val = row['max_value']
            if min_val == '-' or max_val == '-':
                continue
            row['num_value'] = (min_val * max_val) ** 0.5
            km_df.at[index, 'num_value'] = row['num_value']
    ## Now drop the columns min_value and max_value from km_df
    ## check if the columns exist first
    
    if 'min_value' in km_df.columns and 'max_value' in km_df.columns:
        km_df = km_df.drop(columns=['min_value', 'max_value'])
    

    ## make a new column in km_df called change for all rows
    km_df['change'] = '-'
    ## make a new column for uniprot ids in km_df
    km_df['uniprot'] = '-'
    ## iterate over the rows of km_df and assign value to uniprot column
    for index, row in km_df.iterrows():
        km_df.at[index, 'uniprot'] = extract_uniprot_accessions(row['proteins'], proteins)

    ## iterate over the rows of km_df and if the mutant column is not None, then add a new column
    ## called change and assign the value of mutant to it
    for index, row in engg_df.iterrows():  
        for index2, row2 in km_df.iterrows():
            flag11 = False
            if row['proteins'] == row2['proteins'] and row['references'] == row2['references']:
                if flag11 == True:
                    print("Error") ## this should not happen
                    
                if row2['mutation'] != '-':
                    flag11 = True
                    km_df.at[index2, 'change'] = row['value']
            
    # print(km_df.head())
    # print(engg_df.head())
    ## Now save the km_df as a csv file
    km_df.to_csv(f"./cache_dir/{ec_num}-km.csv")
    engg_df.to_csv(f"./cache_dir/{ec_num}-engg.csv")
print("Done, time elapsed: ", time.time()-start)
logger.info("Done with all EC numbers")
logger.info(f"Time elapsed: {time.time()-start}")

logger.info("Closing create.py")

