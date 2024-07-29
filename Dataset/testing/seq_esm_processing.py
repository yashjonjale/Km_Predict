import torch
import esm
import numpy as np

emb_dim = 1260




# Path to the directory containing the model files
model_dir = "/home/yashjonjale/Documents/Dataset/models/esm1_t33_650M_UR50S.pt"

# Load the model
model = esm.Model(model_dir, repr_layers=[33], return_contacts=True)
model.eval()  # Set the model to evaluation mode

# Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)  # Move the model to the GPU if available





def extract_embeddings(lst, model, device):
    # Prepare the input
    batch_converter = model.alphabet.get_batch_converter()
    data = lst
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    # Move the input data to the GPU if available
    batch_tokens = batch_tokens.to(device)

    # Encode the sequences
    with torch.no_grad():
        results = model(batch_tokens)

    # Extract the embeddings
    token_embeddings = results["representations"][33].cpu().numpy()

    # print(token_embeddings.shape)  # This should give you (num_sequences, sequence_length, embedding_dim)
    return token_embeddings





import glob # For file handling
import pandas # For data handling
from Bio import ExPASy, SwissProt # For sequence retrieval



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
    for token in tokens:
        index = int(token[1:-1]) - 1
        from1 = token[0]
        to = token[-1]
        if seq[index] != from1:
            # print(f"Error: {seq[index]} != {from1}")
            return "ignore_entry"
        seq = seq[:index] + to + seq[index+1:]
    return seq







# Path to the directory containing the CSV files
directory = "/home/yashjonjale/Documents/Dataset/cache_dir/"

# Naming format for the CSV files
naming_format = "*sub.csv"

# Get a list of all CSV files in the directory
km_csv_files = glob.glob(f"{directory}/{naming_format}")

# Dictionary to store the substrates and their corresponding IDs

# import json

# with open('./cache_dir/substr2substr_id.json', 'r') as f:
#     substr2substr_id = json.load(f)

dict_esm_mapping = {}
idnum = 0

for fname in km_csv_files:
    km_df = pandas.read_csv(fname)
    print(f"Processing {fname}")
    km_df['seq_final'] = None
    km_df['seq_orig'] = None
    km_df['esm_id'] = None
    new_fname = fname.replace('sub.csv', 'sub_esm.csv')
    lst = []
    for index, row in km_df.iterrows():
        # if row['uniprot'] == "ignore_entry":
        #     km_df.drop(index, inplace=True)
        #     continue

        seq = safe_extract_sequence(row['uniprot'])
        seq1 = modify_seq(seq, row['change'])
        if seq == None:
            ##assuming this does not happen
            print("seq is none")
            break
        if seq1 != "ignore_entry":
            lst.append(seq1)
        else:
            lst.append("")
        
        km_df.at[index, 'seq_orig'] = seq
        km_df.at[index, 'seq_final'] = seq1

    ## extract the embeddings for the sequences in lst
    embeddings = extract_embeddings(lst, model, device)                                                                                                  
    num = len(lst)
    for i in range(num):
        km_df.at[i, 'esm_id'] = idnum
        dict_esm_mapping[idnum] = embeddings[i]
        idnum += 1
    km_df.to_csv(new_fname, index=False)
    print(f"Saved {new_fname}")




