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

path = '/../Dataset/brenda_analyse/'

df = pd.read_csv(os.getcwd()+path+'total_esm.csv')

#sort with respect to EC_ID column
df = df.sort_values(by=['EC_ID'])
#apply ast to the esm column
df['esm'] = df['esm'].apply(ast.literal_eval)
print(df['esm'].dtypes)
#make a column "SEEN" to keep track
df['SEEN'] = 0
#make a column for substrate_ID
df['SUBSTRATE_ID'] = 0
#make a new column called REF
df['REF'] = 0

#make a dict for substrate_id 
substrate_id = {}
inv_substrate_id = {}
for i, substrate in enumerate(df['substrate'].unique()):
    substrate_id[substrate] = i
    inv_substrate_id[i] = substrate

#add the substrate_id to the dataframe
for i in range(len(df)):
    df.loc[i, 'SUBSTRATE_ID'] = substrate_id[df.loc[i, 'substrate']]


#save these dictionaries as json
path = os.getcwd() + '/refined_model/data/'

with open(path + 'substrate_id.json', 'w') as f:
    json.dump(substrate_id, f)

with open(path + 'inv_substrate_id.json', 'w') as f:
    json.dump(inv_substrate_id, f)

for i in range(len(df)):
    if df.loc[i,'change']!='-':
        df.loc[i,'REF']=1

#make a new dataframe with the same columns as the original dataframe
df_train = pd.DataFrame(columns=df.columns)

#make groups with respect to EC_ID and SUBSTRATE_ID
groups = df.groupby(['EC_ID', 'substrate'])



print('Number of groups:', len(groups))
#print average size of the groups
print('Average size of the groups:', len(df)/len(groups))
#initialize the train and test dataframes, these will contain the pairs of entries



num_ref = 0
num_grps = 0
tst = 0
trn = 0
df_train = pd.DataFrame(columns=df.columns)
tn = []
ts = []

for name, group in groups:
    num_grps += 1
    if num_grps % 10 == 0:
        print('Number of groups done:', num_grps)
        
    if len(group)<2:
        continue
    group.reset_index(drop=True, inplace=True)
    subs_name = name
    ref_group = group[group['REF'] == 1]
    non_ref_group = group[group['REF'] == 0]
    ref_group.reset_index(drop=True, inplace=True)
    non_ref_group.reset_index(drop=True, inplace=True)
    prs_train = []
    prs_test = []
    for i in range(len(non_ref_group)):
        print(f"The ratio is {tst/(0.000001+trn)}")
        if tst/(0.000001+trn)>0.25:
            flag = 1
        else:
            flag = 0
        for j in range(len(ref_group)):
            #make a pair
            entry = {}
            entry['EC_ID'] = name[0]
            entry['SUBSTRATE_ID'] = non_ref_group.loc[i, 'SUBSTRATE_ID']
            entry['UNIPROT_MUT'] = non_ref_group.loc[i, 'uniprot']
            entry['UNIPROT_REF'] = ref_group.loc[j, 'uniprot']
            entry['CHANGE_REF'] = ref_group.loc[j, 'change']
            entry['CHANGE_MUT'] = non_ref_group.loc[i, 'change']
            entry['SEQ_REF'] = ref_group.loc[j, 'seq_str']
            entry['SEQ_MUT'] = non_ref_group.loc[i, 'seq_str']
            entry['ESM_REF'] = list(ref_group.loc[j, 'esm'])
            entry['ESM_MUT_delta'] = list(np.array(non_ref_group.loc[i, 'esm']) - np.array(ref_group.loc[j, 'esm']))
            entry['KM_REF'] = ref_group.loc[j, 'num_value_gm']
            entry['KM_MUT'] = non_ref_group.loc[i, 'num_value_gm']
            entry['LABEL'] = "0"
            seq1 = ref_group.loc[j,'seq_str']
            seq2 = non_ref_group.loc[i,'seq_str']
            dist = edit_distance(seq1,seq2)
            sim = 1 - dist/min(len(seq1),len(seq2))
            if sim<0.95:
                continue
            
            if flag:
                prs_train.append(entry)
                trn+=1
            else:
                prs_test.append(entry)
                tst+=1
            
    
        
    for i in range(len(ref_group)):
        print(f"The ratio is {tst/(0.000001+trn)}")
        if tst/(0.000001+trn)>0.25:
            flag = 1
        else:
            flag = 0
        for j in range(i+1,len(ref_group)):
            entry = {}
            entry['EC_ID'] = name[0]
            entry['SUBSTRATE_ID'] = ref_group.loc[i, 'SUBSTRATE_ID']
            entry['UNIPROT_MUT'] = ref_group.loc[i, 'uniprot']
            entry['UNIPROT_REF'] = ref_group.loc[j, 'uniprot']
            entry['CHANGE_REF'] = ref_group.loc[j, 'change']
            entry['CHANGE_MUT'] = ref_group.loc[i, 'change']
            entry['SEQ_REF'] = ref_group.loc[j, 'seq_str']
            entry['SEQ_MUT'] = ref_group.loc[i, 'seq_str']
            entry['ESM_REF'] = list(ref_group.loc[j, 'esm'])
            entry['ESM_MUT_delta'] = list(np.array(ref_group.loc[i, 'esm']) - np.array(ref_group.loc[j, 'esm']))
            entry['KM_REF'] = ref_group.loc[j, 'num_value_gm']
            entry['KM_MUT'] = ref_group.loc[i, 'num_value_gm']
            entry['LABEL'] = "0"
            seq1 = ref_group.loc[j,'seq_str']
            seq2 = ref_group.loc[i,'seq_str']
            dist = edit_distance(seq1,seq2)
            sim = 1 - dist/min(len(seq1),len(seq2))
            if sim<0.95:
                continue
            
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
        
train = pd.DataFrame(columns=['EC_ID','SUBSTRATE_ID','UNIPROT_REF','UNIPROT_MUT','CHANGE_REF','CHANGE_MUT','SEQ_REF','SEQ_MUT','ESM_REF','ESM_MUT_delta', 'KM_REF', 'KM_MUT', 'SIM','LABEL'])
test = pd.DataFrame(columns=['EC_ID','SUBSTRATE_ID','UNIPROT_REF','UNIPROT_MUT','CHANGE_REF','CHANGE_MUT','SEQ_REF','SEQ_MUT','ESM_REF','ESM_MUT_delta', 'KM_REF', 'KM_MUT', 'SIM','LABEL'])

for x in tn:
    for y in x:
        train = train.append(y, ignore_index=True)
for x in ts:
    for y in x:
        test = test.append(y, ignore_index=True)
        

print(f"The train pairs are {len(train)}")
print(f"The test pairs are {len(test)}")

#sort the train and test dataframes with respect to EC_ID and SUBSTRATE_ID
train = train.sort_values(by=['EC_ID', 'SUBSTRATE_ID'])
test = test.sort_values(by=['EC_ID', 'SUBSTRATE_ID'])

#save the train and test dataframes
train.to_csv(path + 'train_pairs.csv', index=False)

test.to_csv(path + 'test_pairs.csv', index=False)

    
    
#Plot a histogram for group sizes
group_sizes = []
for name, group in groups:
    if len(group) > 20:
        continue
    group_sizes.append(len(group))
plt.hist(group_sizes, bins=100)
plt.xlabel('Group size')
plt.ylabel('Frequency')
plt.title('Distribution of group sizes')
plt.savefig(path + 'group_sizes.png')

#p5rint the stats for df_train
print('Number of unique substrates:', len(df_train['SUBSTRATE_ID'].unique()))
print('Number of unique EC_ID:', len(df_train['EC_ID'].unique()))
#entries?
print('Number of entries in train:', len(df_train))


print('Number of unique substrates:', len(substrate_id))
print('Number of unique EC_ID:', len(df['EC_ID'].unique()))
print('Number of unique EC_ID in train:', len(train['EC_ID'].unique()))
print('Number of unique EC_ID in test:', len(test['EC_ID'].unique()))

print('Number of unique substrates in train:', len(train['SUBSTRATE_ID'].unique()))
print('Number of unique substrates in test:', len(test['SUBSTRATE_ID'].unique()))

print('Number of unique pairs in train:', len(train))
print('Number of unique pairs in test:', len(test))

# #plot the distribution of the pairs and save the plot 
# plt.hist(test['KM_MUT']-test['KM_REF'], bins=100)
# plt.xlabel('KM_delta')
# plt.ylabel('Frequency')
# plt.title('Distribution of KM_delta')
# plt.savefig(path + 'KM_delta.png')


print('Done!')



def split_train_test(df):
    #sort with respect to EC_ID and SUBSTRATE_ID column
    df = df.sort_values(by=['EC_ID', 'SUBSTRATE_ID'])
    #group by EC_ID and SUBSTRATE_ID
    groups = df.groupby(['EC_ID', 'SUBSTRATE_ID'])
    
    #intialize the 5 folds
    folds = []
    for i in range(5):
        folds.append(pd.DataFrame(columns=['EC_ID','SUBSTRATE_ID','UNIPROT_REF','UNIPROT_MUT','CHANGE_REF','CHANGE_MUT','SEQ_REF','SEQ_MUT','ESM_REF','ESM_MUT_delta', 'KM_REF', 'KM_MUT','SIM', 'LABEL']))
    
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
                entry['SIM'] = 1 - edit_distance(entry['SEQ_REF'], entry['SEQ_MUT'])/min(len(entry['SEQ_REF']), len(entry['SEQ_MUT']))
                
                
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







def prepare_data(df):
    import ast
    #delete the columns that are not needed
    df = df.drop(columns=['EC_ID','LABEL'])
    df['ESM_REF'] = df['ESM_REF'].apply(ast.literal_eval)  
    df['ESM_MUT_delta'] = df['ESM_MUT_delta'].apply(ast.literal_eval)
    df['KM_REF'] = df['KM_REF'].apply(ast.literal_eval)
    df['KM_MUT'] = df['KM_MUT'].apply(ast.literal_eval)
    df['SEQ_REF_ID'] = df['SEQ_REF_ID'].apply(ast.literal_eval)
    df['SEQ_MUT_ID'] = df['SEQ_MUT_ID'].apply(ast.literal_eval)
    df['SUBSTRATE_ID'] = df['SUBSTRATE_ID'].apply(ast.literal_eval)
    df_ww = df[df['CHANGE_REF'] == '-' and df['CHANGE_MUT'] == '-']
    df_ww.reset_index(drop=True, inplace=True)
    uniq_seq_ids = set()
    for row in df_ww.iterrows():
        uniq_seq_ids.add(row['SEQ_ID_REF'])
        uniq_seq_ids.add(row['SEQ_ID_MUT'])
    
    seq_ids = list(uniq_seq_ids)
    return df, seq_ids






def five_fold_split(df, ref_ids=None):
    import time
    start = time.time()
    # later assume that ec id and substrate id does not exist, and SEQ_REF and SEQ_MUT also do not exist
    # df = df.sort_values(by=['EC_ID', 'SUBSTRATE_ID'])
    #apply ast to ESM columms
    #train = pd.DataFrame(#train = pd.DataFrame(columns=['CHANGE_REF','CHANGE_MUT','SEQ_REF_ID','SEQ_MUT_ID','ESM_REF','ESM_MUT_delta', 'KM_REF', 'KM_MUT', 'LABEL'])columns=['CHANGE_REF','CHANGE_MUT','SEQ_REF_ID','SEQ_MUT_ID','ESM_REF','ESM_MUT_delta', 'KM_REF', 'KM_MUT', 'LABEL'])
    fld_nums = [0,0,0,0,0]    
    #group by EC_ID and SUBSTRATE_ID
    # groups = df.groupby(['EC_ID', 'SUBSTRATE_ID'])
    
    folds = []
    for i in range(5):
        folds.append(pd.DataFrame(columns=['SUBSTRATE_ID','CHANGE_REF','CHANGE_MUT','SEQ_REF_ID','SEQ_MUT_ID','ESM_REF','ESM_MUT_delta', 'KM_REF', 'KM_MUT']))
    
    df_mw = df[df['CHANGE_REF'] != '-' or df['CHANGE_MUT'] != '-']
    df_mw.reset_index(drop=True, inplace=True)
    #make 5 parts of the df_mw
    for i in range(len(df_mw)):
        folds[i%5] = folds[i%5].append(df_mw.loc[i], ignore_index=True)
        fld_nums[i%5] += 1
    
    df_ww = df[df['CHANGE_REF'] == '-' and df['CHANGE_MUT'] == '-']
    df_ww.reset_index(drop=True, inplace=True)
    seq_ids = ref_ids
    for i in range(len(seq_ids)):
        fld_i = i%5
        fld_nums[fld_i] += 1
        for j in range(len(df_ww)):
            if df_ww.loc[j, 'SEQ_ID_MUT'] == seq_ids[i]:
                folds[fld_i] = folds[fld_i].append(df_ww.loc[j], ignore_index=True)
                
    folds_out = []
    for i in range(5):
        #keep only the the esm and km columns, delete the rest
        folds[i] = folds[i].drop(columns=['SEQ_REF', 'SEQ_MUT'])
        esm_ref = np.array(folds[i]['ESM_REF'].tolist())
        esm_mut_del = np.array(folds[i]['ESM_MUT_delta'].tolist())
        km_delta = np.array(folds[i]['KM_MUT'].tolist()) - np.array(folds[i]['KM_REF'].tolist())
        km_ref = np.array(folds[i]['KM_REF'].tolist())
        km_mut = np.array(folds[i]['KM_MUT'].tolist())
        subs_id = np.array(folds[i]['SUBSTRATE_ID'].tolist())
        
        #concatenate the esm columns
        esm = np.concatenate((esm_ref, esm_mut_del, subs_id.reshape((-1,1))), axis=1)
        folds_out.append((esm, km_delta, km_ref, km_mut))
        
        
    print(f"The number of entries in each fold are {len(folds[0])}, {len(folds[1])}, {len(folds[2])}, {len(folds[3])}, {len(folds[4])}")
    print(f"The total number of entries are {len(df)}")
    print(f"The total number of ww pairs are {len(df_ww)}")
    print(f"The total number of mw pairs are {len(df_mw)}")
    print(f"Total time taken is {time.time()-start}")
    return folds_out