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



import torch
import esm
import numpy as np

emb_dim = 1280




# Path to the directory containing the model files
model_dir = "/home/yashjonjale/Documents/Dataset/models/esm1_t33_650M_UR50S.pt"

# Load the model
model = torch.load(model_dir)
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


sequences = [
    "MKTFFVQHLLGSALASTANPLSLRLCNLRAPSRQFQVAIMFSVFYLLLYLGTLLLFLFRRLFGFSL",
    "MKVFFVQHLLGSALASTANPLSLRLCNLRAPSRQFQVAIMFSVFYLLLYLGTLLLFLFRRLFGFSL",
    "MKVFFVQHLLGSALASTANPLSLRLCNLRAPSRQFQVAIMFSVFYLLLYLGTLLLFLFRRLFGFSL"
]

embeddings = extract_embeddings(sequences, model, device)

print(embeddings.shape)  # This should give you (num_sequences, sequence_length, embedding_dim)