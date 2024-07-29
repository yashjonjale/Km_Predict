import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os 
import argparse
from sklearn.cluster import KMeans
import pandas as pd
import ast

parser = argparse.ArgumentParser(description='KMeans clustering')
parser.add_argument('--index', type=int, default=4, help='num cluster index [10,20,50,100,200,300,400,500,600,700,800,900,1000]')
#thats it
args = parser.parse_args()

index = args.index

sizes = [10,20,50,100,200,300,400,500,600,700,800,900,1000]

cnt_pth = os.getcwd()+"/cluster_model/cluster_centres.npy"

sz_arr = np.load(cnt_pth, allow_pickle=True)

centres = sz_arr[index]

num_clusters = sizes[index]

path =  os.getcwd()+"/../Dataset/brenda_analyse/total_esm.csv"
df = pd.read_csv(path)
y = np.array(df['num_value_gm'].tolist())
df['esm'] = df['esm'].apply(ast.literal_eval)
X = np.array(df['esm'].tolist())

#split the data
X, X_test, y, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

#save the test data
np.save(os.getcwd()+f"/cluster_model/test_data.npy", X_test)
np.save(os.getcwd()+f"/cluster_model/test_labels.npy", y_test)


cluster_groups = []

for i in range(centres.shape[0]):
    cluster_groups.append([])
    
for i in range(X.shape[0]):
    distances = np.linalg.norm(centres - X[i], axis=1)
    cluster_groups[np.argmin(distances)].append(i)

num_batches = 10
batch_size = num_clusters//num_batches

#save one file for each cluster but a separate folder for each batch

for i in range(num_batches):
    if i*batch_size >= num_clusters:
        break
    os.makedirs(os.getcwd()+f"/cluster_model/batches/b{i}", exist_ok=True)
    for j in range(batch_size):
        #now make a separate np array to store the data for each cluster
        if i*batch_size+j >= num_clusters:
            break
        lst = []
        for k in cluster_groups[i*batch_size+j]:
            lst.append(np.concatenate((X[k], [y[k]])))
        arr_cluster = np.array(lst)
        np.save(os.getcwd()+f"/cluster_model/batches/b{i}/cluster_{i*batch_size+j}.npy", arr_cluster)
        
