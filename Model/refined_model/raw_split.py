import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import ast
import os

def edit_distance(str1, str2):
    m = len(str1)
    r = len(str2)

    # Create a DP table to store results of subproblems
    dp = [[0 for x in range(r + 1)] for x in range(m + 1)]

    # Fill dp[][] in bottom up manner
    for i in range(m + 1):
        for j in range(r + 1):
            # If first string is empty, only option is to
            # insert all characters of second string
            if i == 0:
                dp[i][j] = j  # Min. operations = j

            # If second string is empty, only option is to
            # remove all characters of first string
            elif j == 0:
                dp[i][j] = i  # Min. operations = i

            # If last characters are the same, ignore the last character
            # and recur for the remaining substring
            elif str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]

            # If the last character is different, consider all
            # possibilities and find the minimum
            else:
                dp[i][j] = 1 + min(dp[i][j - 1],    # Insert
                                   dp[i - 1][j],    # Remove
                                   dp[i - 1][j - 1])  # Replace

    return dp[m][r]

split_ratio = 0.2

path = '/refined_model/'

df = pd.read_csv(os.getcwd()+path+'total_esm.csv')

df['seq_str'] = df['seq_str'].astype(str)
df['substrate'] = df['substrate'].astype(str)
df['esm'] = df['esm'].apply(ast.literal_eval)
df['SUBSTRATE_ID'] = 0

df['REF'] = 0
df['SEQ_ID'] = 0

substrate_id = {}
inv_substrate_id = {}
for i, substrate in enumerate(df['substrate'].unique()):
    substrate_id[substrate] = i
    inv_substrate_id[i] = substrate
    
seq2id = {}
id2seq = {}
for i, seq in enumerate(df['seq_str'].unique()):
    seq2id[seq] = i
    id2seq[i] = seq

print(f"The number of unique substrates is {len(substrate_id.keys())}")
print(f"The number of unique sequences is {len(seq2id.keys())}")

for i in range(len(df)):
    df.loc[i, 'SUBSTRATE_ID'] = int(substrate_id[str(df.loc[i, 'substrate'])])
    df.loc[i, 'SEQ_ID'] = int(seq2id[str(df.loc[i, 'seq_str'])])
    
for i in range(len(df)):
    if df.loc[i,'change']=='-':
        df.loc[i,'REF']=1
        
print(df.dtypes)

#save these dictionaries as json
path = os.getcwd() + '/refined_model/data/'

with open(path + 'substrate_id.json', 'w') as f:
    json.dump(substrate_id, f)

with open(path + 'inv_substrate_id.json', 'w') as f:
    json.dump(inv_substrate_id, f)
    
with open(path + 'seq2id.json', 'w') as f:
    json.dump(seq2id, f)

with open(path + 'id2seq.json', 'w') as f:
    json.dump(id2seq, f)

groups = df.groupby(['EC_ID', 'SUBSTRATE_ID'])

sims = []
mlens = []
dists = []
num_ref = 0
num_grps = 0
tst = 0
trn = 0
tn = []
ts = []

gs = []
ref_lens = []
non_ref_lens = []

for name, group in groups:
    num_grps += 1
    if num_grps % 100 == 0:
        print('Number of groups done:', num_grps)
        
    if len(group)<2:
        continue
    group.reset_index(drop=True, inplace=True)
    subs_name = name
    ref_group = group[group['REF'] == 1]
    non_ref_group = group[group['REF'] == 0]
    ref_group.reset_index(drop=True, inplace=True)
    non_ref_group.reset_index(drop=True, inplace=True)
    ref_lens.append(len(ref_group))
    non_ref_lens.append(len(non_ref_group))
    gs.append(len(group))
    prs_train = []
    prs_test = []
    for i in range(len(non_ref_group)):
        if tst/(0.000001+trn)>0.25:
            flag = 1
        else:
            flag = 0
        for j in range(len(ref_group)):
            entry = {}
            entry['EC_ID'] = name[0]
            entry['SUBSTRATE_ID'] = int(non_ref_group.loc[i, 'SUBSTRATE_ID'])
            entry['CHANGE_REF'] = ref_group.loc[j, 'change']
            entry['CHANGE_MUT'] = non_ref_group.loc[i, 'change']
            entry['SEQ_REF_ID'] = int(ref_group.loc[j, 'SEQ_ID'])
            entry['SEQ_MUT_ID'] = int(non_ref_group.loc[i, 'SEQ_ID'])
            entry['ESM_REF'] = ref_group.loc[j, 'esm']
            entry['ESM_MUT_delta'] = list(np.array(non_ref_group.loc[i, 'esm']) - np.array(ref_group.loc[j, 'esm']))
            entry['KM_REF'] = ref_group.loc[j, 'num_value_gm']
            entry['KM_MUT'] = non_ref_group.loc[i, 'num_value_gm']
            seq1 = ref_group.loc[j,'seq_str']
            seq2 = non_ref_group.loc[i,'seq_str']
            dist = edit_distance(seq1,seq2)
            sim = 1 - (dist/min(len(seq1),len(seq2)))
            entry['SIM'] = sim
            sims.append(sim)
            dists.append(dist)
            mlens.append(min(len(seq1),len(seq2)))
            
            if flag:
                prs_train.append(entry)
                trn+=1
            else:
                prs_test.append(entry)
                tst+=1
            
    if len(prs_test) != 0:
        ts.append(prs_test)
    if len(prs_train) != 0:
        tn.append(prs_train)
    for i in range(len(ref_group)):
        if tst/(0.000001+trn)>0.25:
            flag = 1
        else:
            flag = 0
        for j in range(i+1,len(ref_group)):
            entry = {}
            entry['EC_ID'] = name[0]
            entry['SUBSTRATE_ID'] = int(ref_group.loc[i, 'SUBSTRATE_ID'])
            entry['CHANGE_REF'] = str(ref_group.loc[j, 'change'])
            entry['CHANGE_MUT'] = str(ref_group.loc[i, 'change'])
            entry['SEQ_REF_ID'] = int(ref_group.loc[j, 'SEQ_ID'])
            entry['SEQ_MUT_ID'] = int(ref_group.loc[i, 'SEQ_ID'])
            entry['ESM_REF'] = ref_group.loc[j, 'esm']
            entry['ESM_MUT_delta'] = list(np.array(ref_group.loc[i, 'esm']) - np.array(ref_group.loc[j, 'esm']))
            entry['KM_REF'] = ref_group.loc[j, 'num_value_gm']
            entry['KM_MUT'] = ref_group.loc[i, 'num_value_gm']
            seq1 = ref_group.loc[j,'seq_str']
            seq2 = ref_group.loc[i,'seq_str']
            dist = edit_distance(seq1,seq2)
            sim = 1 - (dist/min(len(seq1),len(seq2)))
            sims.append(sim)
            dists.append(dist)
            mlens.append(min(len(seq1),len(seq2)))                       
            if flag:
                prs_train.append(entry)
                trn+=1
            else:
                prs_test.append(entry)
                tst+=1
    if len(prs_test) != 0:
        ts.append(prs_test)
    if len(prs_train) != 0:
        tn.append(prs_train)   

train = pd.DataFrame(columns=['EC_ID','SUBSTRATE_ID','CHANGE_REF','CHANGE_MUT','SEQ_REF_ID','SEQ_MUT_ID','ESM_REF','ESM_MUT_delta', 'KM_REF', 'KM_MUT','SIM'])
test = pd.DataFrame(columns=['EC_ID','SUBSTRATE_ID','CHANGE_REF','CHANGE_MUT','SEQ_REF_ID','SEQ_MUT_ID','ESM_REF','ESM_MUT_delta', 'KM_REF', 'KM_MUT','SIM'])

for x in tn:
    for y in x:
        train = train.append(y, ignore_index=True)
for x in ts:
    for y in x:
        test = test.append(y, ignore_index=True)
        

print(f"The train pairs are {len(train)}")
print(f"The test pairs are {len(test)}")

#save the train and test dataframes
train.to_csv(path + 'train_pairs_idx.csv', index=False)

test.to_csv(path + 'test_pairs_idx.csv', index=False)

#Plot a histogram for group sizes

# plt.hist(gs, bins=100)
# plt.xlabel('Group size')
# plt.ylabel('Frequency')
# plt.title('Distribution of group sizes')
# plt.savefig(path + 'group_sizes.png')

# plt.hist(ref_lens, bins=100)
# plt.xlabel('Ref Group size')
# plt.ylabel('Frequency')
# plt.title('Distribution of ref group sizes')
# plt.savefig(path + 'ref_group_sizes.png')

# plt.hist(gs, bins=100)
# plt.xlabel('Non Ref Group size')
# plt.ylabel('Frequency')
# plt.title('Distribution of non ref group sizes')
# plt.savefig(path + 'non_ref_group_sizes.png')

arr = np.array(dist)/np.array(mlens)
sim_ = 1 - arr

plt.hist(sims, bins=100)
plt.xlabel('Similarity')
plt.ylabel('Frequency')
plt.title('Distribution of sequence identities')
plt.savefig(path + 'sims.png')

bad = 0
for i in range(len(gs)):
    if gs[i]>1:
        if ref_lens[i]==0:
            bad+=1
print(f"Number of groups with no ref group is {bad}")

print('Number of unique EC_ID:', len(df['EC_ID'].unique()))
print('Number of unique EC_ID in train:', len(train['EC_ID'].unique()))
print('Number of unique EC_ID in test:', len(test['EC_ID'].unique()))

print('Number of unique substrates in train:', len(train['SUBSTRATE_ID'].unique()))
print('Number of unique substrates in test:', len(test['SUBSTRATE_ID'].unique()))

print('Number of unique pairs in train:', len(train))
print('Number of unique pairs in test:', len(test))


def prepare_data(path):
    import ast
    df = pd.read_csv(path)
#     print(df.columns)
#     print(df.dtypes)
#     print(df['SEQ_REF_ID'].dtype)
#     print(type(df.loc[0,'SEQ_REF_ID']))
    #delete the columns that are not needed
    df = df.drop(columns=['EC_ID'])
    df['ESM_REF'] = df['ESM_REF'].apply(ast.literal_eval)  
    df['ESM_MUT_delta'] = df['ESM_MUT_delta'].apply(ast.literal_eval)
    df['KM_REF'] = df['KM_REF'].astype(float)
    df['SEQ_REF_ID'] = df['SEQ_REF_ID'].astype(int)
    df['SEQ_MUT_ID'] = df['SEQ_MUT_ID'].astype(int)
    df['SUBSTRATE_ID'] = df['SUBSTRATE_ID'].astype(int)
    df['SIM'] = df['SIM'].astype(float)
    # delete the rows where SIM is less than 0.9
    df = df[df['SIM'] >= 0.3]
    df['SEQ_REF_ID'] = df['SEQ_REF_ID'].apply(lambda x: int(x))
    df['SEQ_MUT_ID'] = df['SEQ_MUT_ID'].apply(lambda x: int(x))
    #drop the SIM column
    df = df.drop(columns=['SIM'])
    df_ww = df[(df['CHANGE_REF'] == '-') & (df['CHANGE_MUT'] == '-')]
    df_ww.reset_index(drop=True, inplace=True)
    uniq_seq_ids = set()
    for i in range(len(df_ww)):
        uniq_seq_ids.add(df_ww.loc[i, 'SEQ_REF_ID'])
        uniq_seq_ids.add(df_ww.loc[i, 'SEQ_MUT_ID'])
    seq_ids = list(uniq_seq_ids)
    return df, seq_ids

path = os.getcwd() + '/refined_model/data/train_pairs_idx.csv'
df1, ref_ids = prepare_data(path)


def five_fold_split(df, ref_ids=None):
    import time
    start = time.time()
    # later assume that ec id and substrate id does not exist, and SEQ_REF and SEQ_MUT also do not exist
    # df = df.sort_values(by=['EC_ID', 'SUBSTRATE_ID'])
    #apply ast to ESM columms
    #train = pd.DataFrame(columns=['CHANGE_REF','CHANGE_MUT','SEQ_REF_ID','SEQ_MUT_ID','ESM_REF','ESM_MUT_delta', 'KM_REF', 'KM_MUT', 'LABEL'])
    fld_nums = [0,0,0,0,0]    
    #group by EC_ID and SUBSTRATE_ID
    # groups = df.groupby(['EC_ID', 'SUBSTRATE_ID'])
    
    folds = []
    for i in range(5):
        folds.append(pd.DataFrame(columns=['SUBSTRATE_ID','CHANGE_REF','CHANGE_MUT','SEQ_REF_ID','SEQ_MUT_ID','ESM_REF','ESM_MUT_delta', 'KM_REF', 'KM_MUT']))
    
    df_mw = df[(df['CHANGE_REF'] != '-') | (df['CHANGE_MUT'] != '-')]
    df_mw.reset_index(drop=True, inplace=True)
    #make 5 parts of the df_mw
    for i in range(len(df_mw)):
        folds[i%5] = folds[i%5].append(df_mw.loc[i], ignore_index=True)
        fld_nums[i%5] += 1
    
    df_ww = df[(df['CHANGE_REF'] == '-') & (df['CHANGE_MUT'] == '-')]
    df_ww.reset_index(drop=True, inplace=True)
    seq_ids = ref_ids
    for i in range(len(seq_ids)):
        fld_i = i%5
        fld_nums[fld_i] += 1
        for j in range(len(df_ww)):
            if df_ww.loc[j, 'SEQ_MUT_ID'] == seq_ids[i]:
                folds[fld_i] = folds[fld_i].append(df_ww.loc[j], ignore_index=True)
                
    folds_out = []
    for i in range(5):
        #keep only the the esm and km columns, delete the rest
        esm_ref = np.array(folds[i]['ESM_REF'].tolist())
        esm_mut_del = np.array(folds[i]['ESM_MUT_delta'].tolist())
        km_delta = np.array(folds[i]['KM_MUT'].tolist()) - np.array(folds[i]['KM_REF'].tolist())
        km_ref = np.array(folds[i]['KM_REF'].tolist())
        km_mut = np.array(folds[i]['KM_MUT'].tolist())
        subs_id = np.array(folds[i]['SUBSTRATE_ID'].tolist())
        # print(f"The shape of esm_ref is {esm_ref.shape}")
        # print(f"The shape of esm_mut_del is {esm_mut_del.shape}")
        # print(f"The shape of km_delta is {km_delta.shape}")
        # print(f"The shape of km_ref is {km_ref.shape}")
        # print(f"The shape of km_mut is {km_mut.shape}")
        # print(f"The shape of subs_id is {subs_id.shape}")
        
        
        #concatenate the esm columns
        esm = np.concatenate((esm_ref, esm_mut_del, subs_id.reshape((-1,1))), axis=1)
        folds_out.append((esm, km_delta, km_ref, km_mut))
        
        
    # print(f"The number of entries in each fold are {len(folds[0])}, {len(folds[1])}, {len(folds[2])}, {len(folds[3])}, {len(folds[4])}")
    # print(f"The total number of entries are {len(df)}")
    # print(f"The total number of ww pairs are {len(df_ww)}")
    # print(f"The total number of mw pairs are {len(df_mw)}")
    # print(f"Total time taken is {time.time()-start}")
    return folds_out

folds = five_fold_split(df1,ref_ids)

#okay, so no issues, print that
print("No issues")
print(f"Done")