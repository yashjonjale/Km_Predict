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
import glob
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

parser = argparse.ArgumentParser(description='KMeans clustering')
parser.add_argument('--index', type=int, default=5, help='num cluster index [10,20,50,100,200,300,400,500,600,700,800,900,1000]')
#thats it
args = parser.parse_args()

index = args.index

sizes = [10,20,50,100,200,300,400,500,600,700,800,900,1000]

cnt_pth = os.getcwd()+"/cluster_model/cluster_centres.npy"

sz_arr = np.load(cnt_pth, allow_pickle=True)

centres = sz_arr[index]
print(f"Centres shape: {centres.shape}")
num_clusters = sizes[index]

# path =  os.getcwd()+"/../Dataset/brenda_analyse/total_esm.csv"
# df = pd.read_csv(path)
# y = np.array(df['num_value_gm'].tolist())
# df['esm'] = df['esm'].apply(ast.literal_eval)
# X = np.array(df['esm'].tolist())

#load the test data
X_test = np.load(os.getcwd()+f"/cluster_model/test_data.npy")
y_test = np.load(os.getcwd()+f"/cluster_model/test_labels.npy")

# data_group_train = []
data_group_test = []

# for i in range(X.shape[0]):
#     distances = np.linalg.norm(centres - X[i], axis=1)
#     data_group_train.append(np.argmin(distances))

for j in range(X_test.shape[0]):
    distances = np.linalg.norm(centres - X_test[j], axis=1)
    data_group_test.append(np.argmin(distances))

#print unique values
# print(f"Unique values in the training data: {np.unique(data_group_test)}")

#Extract the xgboost models
xgb_models = {}

path = os.getcwd()+f"/cluster_model/batches/b*/xgb_models.npy"

files = glob.glob(path)
files.sort()
num=0
for file in files:
    #load the model
    batch_num = num
    #load the npy file with pickle 
    model = np.load(file, allow_pickle=True)
    for i in range(model.shape[0]):
        # if model[i,0] > (batch_num+1)*20 or model[i,0] < batch_num*20:
        #     # print(f"Error with model number {model[i,0]}")
        #     continue
        xgb_models[model[i,0]] = {}
        xgb_models[model[i,0]]['model'] = model[i,2]
        xgb_models[model[i,0]]['centre'] = model[i,1]
        xgb_models[model[i,0]]['rmse'] = model[i,3]
        xgb_models[model[i,0]]['r2'] = model[i,4]    
    num+=1

print(f"The keys in the xgb_models are {xgb_models.keys()}")


#predict the test data
y_pred = []
for i in range(X_test.shape[0]):
    cluster = data_group_test[i]
    try:
        if xgb_models[cluster]['model'] is None:
            y_pred.append(None)
            continue
        y_pred.append(xgb_models[cluster]['model'].predict(X_test[i].reshape(1,-1))[0])
    except:
        y_pred.append(None)
        print(f"Error with cluster {cluster}")
        print(distances.shape)
        # print(distances)
        
    
yt = []
yt_pred = []
for i in range(len(y_pred)):
    if y_pred[i] is not None:
        yt.append(y_test[i])
        yt_pred.append(y_pred[i])    

#calculate the r2 score
r2 = r2_score(yt, yt_pred)
print(f"Results for the test set!")
print(f"R2 score: {r2}")

#store the results of y_pred and y_test
np.save(os.getcwd()+f"/cluster_model/y_pred_fil.npy", yt_pred)
np.save(os.getcwd()+f"/cluster_model/y_test_fil.npy", yt)
#make a scatter plot of the results
plt.scatter(yt, yt_pred)
plt.xlabel("True values")
plt.ylabel("Predicted values")
plt.title("Predicted vs True values")
plt.savefig(os.getcwd()+f"/cluster_model/predicted_vs_true.png")