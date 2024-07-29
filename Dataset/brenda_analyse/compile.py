import json
from Bio import ExPASy
from Bio import SwissProt
import pandas
from scipy.stats import gmean
from matplotlib import pyplot as plt
import sys
import logging
import logging.handlers
import os


def open_as_dict(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
    
brenda_path = os.getcwd()+'/brenda_2023_1.json'

Brenda_dict = open_as_dict(brenda_path)

ec_nums = list(Brenda_dict['data'].keys())
data = []
 
def get_imp_stuff(Brenda_dict, ec_num):
    if ec_num not in Brenda_dict['data']:
        return None, None, None, None, None
    if Brenda_dict['data'][ec_num] is []:
        return None, None, None, None, None
    if 'km_value' not in Brenda_dict['data'][ec_num].keys():
        return None, None, None, None, None
    if 'proteins' not in Brenda_dict['data'][ec_num].keys():
        return None, None, None,None,None
    else:
        proteins = Brenda_dict['data'][ec_num]['proteins']
    if 'engineering' not in Brenda_dict['data'][ec_num].keys():
        engg = None
    else:
        engg = Brenda_dict['data'][ec_num]['engineering']
    # if 'commment' not in Brenda_dict['data'][ec_num]['engineering'][0]:
        # print(Brenda_dict['data'][ec_num])
        # return None, None, None, None, None
    km_vals = Brenda_dict['data'][ec_num]['km_value']
    references = Brenda_dict['data'][ec_num]['references']
    organisms = Brenda_dict['data'][ec_num]['organisms']
    return km_vals, engg, references, organisms, proteins



def fix_km_val_refs(km_vals):
    new_km_vals = []
    for km_val in km_vals:
        if 'comment' not in km_val.keys():
            refnum = km_val['references'][0]
            if 'proteins' not in km_val.keys():
                if 'organisms' not in km_val.keys():
                    continue
                else:
                    pronum = km_val['organisms'][0]
            else:
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
    if engg is None:
        return None
    new_engg = []
    for engg_val in engg:
        # if 'proteins' not in engg_val.keys():
        #     continue
        if 'comment' not in engg_val.keys():
            # print(engg_val)
            # input("______")
            refnum = engg_val['references'][0]
            if 'proteins' not in engg_val.keys():
                if 'organisms' not in engg_val.keys():
                    continue
                else:
                    pronum = engg_val['organisms'][0]
            else:
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


def extract_uniprot_accessions(id, proteins):
    if id not in proteins.keys():
        return "ignore_entry"
    try:
        if 'source' in proteins[id][0] and proteins[id][0]['source'] == 'uniprot':
            return proteins[id][0]['accessions'][0]
        else:
            return "ignore_entry" ##implies ignore
    except:
        print(f"The protein id {id} is {proteins[id]}")
        return "ignore_entry"


def geometric_mean(series):
    return gmean(series)

print("Done loading the Brenda data")

print(f"Total EC numbers: {len(ec_nums)}")
num_substrate = []
num_no_uni=0
no_pro=0
## start measuring time
import time
start = time.time()
i=0
for ec_num in ec_nums:
    i+=1
    if i%10 == 0:
        print(f"Done with {i} EC numbers and time elapsed is {time.time()-start} seconds")
    
    if len(Brenda_dict['data'][ec_num]) == 0:
        continue
    km_vals, engg, references, organisms, proteins = get_imp_stuff(Brenda_dict, ec_num)
    # if engg is None:##not necessarily a problem
    #     continue
    if proteins is None:
        no_pro+=1
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
    if engg is not None:
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
        if row['num_value'] == '-':
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
    # if len(engg_df) > 0:
    #     for index, row in engg_df.iterrows():  
    #         for index2, row2 in km_df.iterrows():
    #             if row['proteins'] == row2['proteins'] and row['references'] == row2['references']:
    #                 if row2['mutation'] != '-':
    #                     if row2['change'] != '-':
    #                         #delete all rows in km_df with this protein and reference
    #                         for index3, row3 in km_df.iterrows():
    #                             if row3['proteins'] == row['proteins'] and row3['references'] == row['references'] and row3['mutation'] != '-':
    #                                 km_df.drop(index3, inplace=True)
    #                         break
    #                     km_df.at[index2, 'change'] = row['value']
    
    # it = 0
    if len(engg_df) > 0:
        for index, row in engg_df.iterrows():  
            for index2, row2 in km_df.iterrows():
                if row['proteins'] == row2['proteins'] and row['references'] == row2['references']:
                    if row2['mutation'] != '-':
                        if row2['change'] != '-':
                            continue
                        comm = row2['comment']
                        change = row['value']
                        if change in comm and change+"/" not in comm and "/"+change not in comm:
                            # if it < 10:
                            #     print(row2)
                            #     print(comm)
                            #     print(change)
                            #     it+=1
                            #     print("Locked")
                            km_df.at[index2, 'change'] = row['value']
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
    if km_df.empty:
        #print the ec id that it has no uniprots
        num_no_uni+=1
        continue
    #rename the 'value' column to 'substrate'
    km_df.rename(columns={'value':'substrate'}, inplace=True)
    # apply on the entire mutation column
    # print(km_df['substrate'].unique())
    # km_df['change'] = km_df['change'].apply(lambda x: x.split('/'))
    #save the km_df to a csv file

    km_df['num_value'] = pandas.to_numeric(km_df['num_value'], errors='coerce')
    #print the uniqu values int he substrate column
    grouped = km_df.groupby(['proteins','substrate','change','mutation','uniprot']).agg(
        num_value_gm = ('num_value', geometric_mean),
        num_value_am = ('num_value', 'mean'),
    ).reset_index()
    #make new column for ec number
    grouped['EC_ID'] = ec_num
    #save the grouped dataframe to a csv file
    grouped.to_csv(f"./ec_grps/{ec_num}_new.csv", index=False)
    num_substrate.append(len(grouped['substrate'].unique()))
print("Done, time elapsed: ", time.time()-start)
print(f"Number of EC numbers with no uniprots: {num_no_uni}")
print(f"Number of EC numbers with no proteins: {no_pro}")

#print all stats for number of substrates
print(f"Total number of substrates: {sum(num_substrate)}")
print(f"Average number of substrates per EC number: {sum(num_substrate)/len(num_substrate)}")
print(f"Min number of substrates per EC number: {min(num_substrate)}")
print(f"Max number of substrates per EC number: {max(num_substrate)}")
print(f"Number of EC numbers with no substrates: {num_substrate.count(0)}")
print(f"Number of EC numbers with less than 1 substrate: {num_substrate.count(1)}")
print(f"Number of EC numbers with more than 1 substrate: {len(num_substrate) - num_substrate.count(0) - num_substrate.count(1)}")

#make a histogram of the number of substrates
import matplotlib.pyplot as plt
#with 20 bins
plt.hist(num_substrate, bins=50)

plt.show()