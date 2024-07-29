import pandas as pd
import os
import time
import esm
import torch
import numpy as np


file = os.getcwd() + "/brenda_analyse/total.csv"
print("loading file")
print(file)
df = pd.read_csv(file)
#make new column
df['esm'] = None
df['label'] = '-'


# Load ESM-1b model
cwd = os.getcwd()
cwd = str(cwd)
model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_location = cwd+"/model/esm1b_t33_650M_UR50S.pt")
batch_converter = alphabet.get_batch_converter()
device = torch.device("cuda")
model = model.to(device)  # Move the model to the GPU if available
model.eval()
print("Model loaded")


# Extract ESM embeddings
data = []
lst = []
lab = []
bad = 0
for index, row in df.iterrows():
    seq = row['seq_str']
    if len(seq) > 1000:
        seq = "AAAAAAAA"
        bad += 1
    data.append((row['uniprot'] + "|" +row['change'], seq))

print(f"Found {bad} sequences longer than 1000 amino acids")

# Create batches
batch_size = 32
batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
print(f"Processing {len(data)} sequences split into {len(batches)} batches")
j=0
# Extract ESM embeddings
for batch in batches:
    j+=1
    if j%1 == 0:
        print(f"Processing batch {j}/{len(batches)}")
    # Prepare data
    try:
        batch_labels, batch_strs, batch_tokens = batch_converter(batch)
        batch_tokens = batch_tokens.to(device)
    except:
        print("Error with batch")
        print(batch)
        break
    # Extract ESM embeddings
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)

    # Extract the final layer
    # seqs = batch_converter(batch_strs, to_tokens=False)
    # for i, (_, seq) in enumerate(seqs.items()):
    esm_vec = results["representations"][33].cpu().numpy()
    mean_vec = np.mean(esm_vec[:,1:-1,:], axis=1)
    for i in range(esm_vec.shape[0]):
        # df.loc[df['uniprot'] == batch_labels[i].split('|')[0] and df['change'] == batch_labels[i].split('|')[1], 'esm'] = mean_vec
        lst.append(mean_vec[i])
        lab.append(batch_labels[i])

print("Done with the ESM embeddings")
print(len(lst))
print(len(lab))
#update the first 32 rows with the new lists
for i in range(len(lst)):
    df.at[i, 'esm'] = lst[i].tolist()
    df.at[i, 'label'] = lab[i]



df.to_csv(os.getcwd()+"/brenda_analyse/total_esm.csv", index=False)


