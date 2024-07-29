import pandas as pd
import glob
import os
from Bio import ExPASy, SwissProt # For sequence retrieval
import time


def extract_sequence(uniprot_id):
    handle = ExPASy.get_sprot_raw(uniprot_id)
    record = SwissProt.read(handle)
    return record.sequence
def safe_extract_sequence(uniprot_id):
    try:
        return extract_sequence(uniprot_id)
    except ValueError:
        return None  # or '', or any default value you want to use

def modify_seq(seq, change):
    if seq is None:
        return "ignore_entry"
    tokens = change.split('/')
    try:
        for token in tokens:
            index = int(token[1:-1]) - 1
            from1 = token[0]
            to = token[-1]
            if seq[index] != from1:
                # print(f"Error: {seq[index]} != {from1}")
                return "ignore_entry"
            seq = seq[:index] + to + seq[index+1:]
        return seq
    except:
        return "ignore_entry"


# Read all files in the directory
path = "/home/yashjonjale/Documents/intern_proj/Dataset/brenda_analyse/ec_grps"

all_files = glob.glob(os.path.join(path, "*.csv"))
all_files.sort()
print(f"Total files: {len(all_files)}")
print(os.getcwd())
start = time.time() 
uni2seq = {}
i = 0
    
for file in all_files[(870+550+50):]:
    df = pd.read_csv(file)
    df['seq_str'] = '-'
    for index, row in df.iterrows():
        uniprot_id = row['uniprot']
        # print(f"Processing {uniprot_id} and change {row['change']}")
        if uniprot_id not in uni2seq:
            seq = safe_extract_sequence(uniprot_id)
            uni2seq[uniprot_id] = seq
        seq = uni2seq[uniprot_id]
        if seq is None:
            df.drop(index, inplace=True)
            continue
        if len(seq) > 1000:
            df.drop(index, inplace=True)
            continue        
        
        if row['change'] != '-':
            seq_final = modify_seq(seq, row['change'])
            if seq_final == "ignore_entry":
                df.drop(index, inplace=True)
                continue
            df.at[index, 'seq_str'] = seq_final
        else:
            df.at[index, 'seq_str'] = seq
    ec_num = '.'.join(file.strip().split('/')[-1].split('.')[:-1]).split('_')[0]
    # print(file)
    # print(ec_num)
    df.to_csv(os.getcwd()+f"/seqs/{ec_num}.csv")
    i+=1
    if i%10 == 0:
        # logger.info(f"Done {i} files, time elapsed: {time.time()-start}")
        print(f"Done {i} files, time elapsed: ", time.time()-start)            
        
print("Done, time elapsed: ", time.time()-start)
# logger.info(f"Done, time elapsed: {time.time()-start}")

# logger.info("Done loading the Dataset")
    
#save uni2seq as json
import json
with open('uni2seq.json', 'w') as f:
    json.dump(uni2seq, f)