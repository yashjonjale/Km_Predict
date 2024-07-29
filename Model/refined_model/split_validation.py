import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import ast
split_ratio = 0.2


#lets define a custom splitting function

def split_train_test(df):
    #sort with respect to EC_ID and SUBSTRATE_ID column
    df = df.sort_values(by=['EC_ID', 'SUBSTRATE_ID'])
    #group by EC_ID and SUBSTRATE_ID
    groups = df.groupby(['EC_ID', 'SUBSTRATE_ID'])
    
    #intialize the 5 folds
    folds = []
    for i in range(5):
        folds.append(pd.DataFrame(columns=['EC_ID','SUBSTRATE_ID','UNIPROT_REF','UNIPROT_MUT','CHANGE_REF','CHANGE_MUT','SEQ_REF','SEQ_MUT','ESM_REF','ESM_MUT_delta', 'KM_REF', 'KM_MUT', 'LABEL']))
    
    #iterate over the groups and make 2 splits based on the REF column if it is 1 or 0
    for name, group in groups:
        #now make 2 new groups based on the REF column
        ref_group = group[group['REF'] == 1]
        non_ref_group = group[group['REF'] == 0]
        
        #iterate over the ref_group and mut_group and split them into 5 folds
        prs = []
        for i in range(5):
            prs.append([])
        
        for i in range(len(non_ref_group)):
            for j in range(len(ref_group)):
                #make a pair
                entry = {}
                entry['EC_ID'] = non_ref_group.loc[i, 'EC_ID']
                entry['SUBSTRATE_ID'] = non_ref_group.loc[i, 'SUBSTRATE_ID']
                entry['UNIPROT_MUT'] = non_ref_group.loc[i, 'uniprot']
                entry['UNIPROT_REF'] = ref_group.loc[j, 'uniprot']
                entry['CHANGE_REF'] = ref_group.loc[j, 'change']
                entry['CHANGE_MUT'] = non_ref_group.loc[i, 'change']
                entry['SEQ_REF'] = ref_group.loc[j, 'seq_str']
                entry['SEQ_MUT'] = non_ref_group.loc[i, 'seq_str']
                entry['ESM_REF'] = ref_group.loc[j, 'esm']
                entry['ESM_MUT_delta'] = non_ref_group.loc[i, 'esm'] - ref_group.loc[j, 'esm']
                entry['KM_REF'] = ref_group.loc[j, 'num_value_gm']
                entry['KM_MUT'] = non_ref_group.loc[i, 'num_value_gm']
                entry['LABEL'] = "0"
                
                
                if i < 0.2*len(non_ref_group):
                    prs[0].append(entry)
                elif i < 0.4*len(non_ref_group):
                    prs[1].append(entry)
                elif i < 0.6*len(non_ref_group):
                    prs[2].append(entry)
                elif i < 0.8*len(non_ref_group):
                    prs[3].append(entry)
                else:
                    prs[4].append(entry)
                
        #split the ref_group into 5 folds
        prs_ref = []
        for i in range(5):
            prs_ref.append([])
        
        #divide the ref_group entries equally into 5 folds, and make pairs within these folds
        for i in range(len(ref_group)):
            for j in range(len(ref_group)):
                #make a pair
                entry = {}
                entry['EC_ID'] = ref_group.loc[i, 'EC_ID']
                entry['SUBSTRATE_ID'] = ref_group.loc[i, 'SUBSTRATE_ID']
                entry['UNIPROT_MUT'] = ref_group.loc[i, 'uniprot']
                entry['UNIPROT_REF'] = ref_group.loc[j, 'uniprot']
                entry['CHANGE_REF'] = ref_group.loc[j, 'change']
                entry['CHANGE_MUT'] = ref_group.loc[i, 'change']
                entry['SEQ_REF'] = ref_group.loc[j, 'seq_str']
                entry['SEQ_MUT'] = ref_group.loc[i, 'seq_str']
                entry['ESM_REF'] = ref_group.loc[j, 'esm']
                entry['ESM_MUT_delta'] = ref_group.loc[i, 'esm'] - ref_group.loc[j, 'esm']
                entry['KM_REF'] = ref_group.loc[j, 'num_value_gm']
                entry['KM_MUT'] = ref_group.loc[i, 'num_value_gm']
                entry['LABEL'] = "1"
                
                if i < 0.2*len(ref_group):
                    if j < 0.2*len(ref_group):
                        prs_ref[0].append(entry)
                elif i < 0.4*len(ref_group):
                    if j < 0.4*len(ref_group) and j >= 0.2*len(ref_group):
                        prs_ref[1].append(entry)
                elif i < 0.6*len(ref_group):
                    if j < 0.6*len(ref_group) and j >= 0.4*len(ref_group):
                        prs_ref[2].append(entry)
                elif i < 0.8*len(ref_group):
                    if j < 0.8*len(ref_group) and j >= 0.6*len(ref_group):
                        prs_ref[3].append(entry)
                else:
                    if j >= 0.8*len(ref_group):
                        prs_ref[4].append(entry)
                    
        #append prs and prs_ref to the folds
        for i in range(5):
            folds[i] = folds[i].append(prs[i], ignore_index=True)
            folds[i] = folds[i].append(prs_ref[i], ignore_index=True)
    folds_out = []
    for i in range(5):
        #keep only the the esm and km columns, delete the rest
        folds[i] = folds[i].drop(columns=['EC_ID', 'UNIPROT_REF', 'UNIPROT_MUT', 'CHANGE_REF', 'CHANGE_MUT', 'SEQ_REF', 'SEQ_MUT'])
        #convert the esm column to a list using ast
        folds[i]['ESM_REF'] = folds[i]['ESM_REF'].apply(lambda x: ast.literal_eval(x))
        folds[i]['ESM_MUT_delta'] = folds[i]['ESM_MUT_delta'].apply(lambda x: ast.literal_eval(x))  
        esm_ref = np.array(folds[i]['ESM_REF'].tolist())
        esm_mut = np.array(folds[i]['ESM_MUT_delta'].tolist())
        #convert the km column to a list using ast
        folds[i]['KM_REF'] = folds[i]['KM_REF'].apply(lambda x: ast.literal_eval(x))
        folds[i]['KM_MUT'] = folds[i]['KM_MUT'].apply(lambda x: ast.literal_eval(x))
        km_delta = np.array(folds[i]['KM_MUT'].tolist()) - np.array(folds[i]['KM_REF'].tolist())
        km_ref = np.array(folds[i]['KM_REF'].tolist())
        km_mut = np.array(folds[i]['KM_MUT'].tolist())
        #concatenate the esm columns
        esm = np.concatenate((esm_ref, esm_mut), axis=1)
        folds_out.append((esm, km_delta, km_ref, km_mut))
            
    return folds_out
