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
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, rand
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


parser = argparse.ArgumentParser(description='KMeans clustering')
parser.add_argument('--batch', type=int, default=0, help='batch number [0-9]')
parser.add_argument('--index', type=int, default=4, help='num cluster index [10,20,50,100,200,300,400,500,600,700,800,900,1000]')
args = parser.parse_args()

batch = args.batch
index = args.index


#load the cluster centres
cnt_pth = os.getcwd()+"/cluster_model/cluster_centres.npy"
sz_arr = np.load(cnt_pth, allow_pickle=True)
centres = sz_arr[index]



#load the data

path =  os.getcwd()+f"/cluster_model/batches/b{batch}/"
files = glob.glob(path+"cl*.npy")

xgb_models = []


files.sort()
print(f"The number of files in the batch is {len(files)}")
print(files)

val_r2s = []
val_losses = []

import time
for file in files:
    print(f"Processing file {file}")
    start = time.time()
    model_num = int(file.split("/")[-1].split(".")[0].split("_")[-1])
    print(f"Processing model {model_num}")
    centre = centres[int(model_num)]
    data = np.load(file)
    X = data[:,:1280]
    y = data[:,1280:]
    y.reshape(-1)
    if X.shape[0] < 10:
        xgb_models.append((model_num,centre,None, None, None))
        val_r2s.append(0)
        val_losses.append(0)
        continue
    #split X and y into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    
    # #train a xgb model for a dataset with 1280 features and continuous labels with hyperopt tuning
    space = {
        'max_depth': hp.quniform('max_depth', 3, 10, 1),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
        'gamma': hp.uniform('gamma', 0, 1),
        # 'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
        # 'subsample': hp.uniform('subsample', 0.5, 1),
        # 'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
        # 'reg_alpha': hp.uniform('reg_alpha', 0, 1),  # L1 regularization
        # 'reg_lambda': hp.uniform('reg_lambda', 0, 1)  # L2 regularization
        # "learning_rate": hp.uniform("learning_rate", 0.01, 1),
        # "max_depth": hp.uniform("max_depth", 4,12),
        #"subsample": hp.uniform("subsample", 0.7, 1),
        "reg_lambda": hp.uniform("reg_lambda", 0, 5),
        "reg_alpha": hp.uniform("reg_alpha", 0, 5),
        "max_delta_step": hp.uniform("max_delta_step", 0, 5),
        "min_child_weight": hp.uniform("min_child_weight", 0.1, 15),
        "num_rounds":  hp.uniform("num_rounds", 20, 400)
    }
    def objective(params):
        rounds = int(params['num_rounds'])
        params = {
            'gamma': params['gamma'],
            'max_depth': int(np.round(params['max_depth'])),
            'learning_rate': params['learning_rate'],
            'min_child_weight': params['min_child_weight'],
            'reg_alpha': params['reg_alpha'],  # L1 regularization
            'reg_lambda': params['reg_lambda'],  # L2 regularization
            'tree_method': 'gpu_hist',  # Use CPU for training        
            'predictor': 'gpu_predictor',  # Use GPU for predictions
            'sampling_method': 'uniform',
            'eval_metric': 'rmse'
        }
        model = xgb.train(params, xgb.DMatrix(X_train,label=y_train),rounds)
        y_pred = model.predict(xgb.DMatrix(X_val))
        return {'loss': np.mean((y_val-y_pred)**2), 'status': STATUS_OK}
    trials = Trials()
    best = fmin(fn = objective, space = space,
            algo=rand.suggest, max_evals = 100, trials=trials,verbose=False)
    print(f"Best hyperparameters for {file}: {best}")
    rounds1 = int(np.round(best['num_rounds']))
    del best['num_rounds']
    params={'max_depth': int(np.round(best['max_depth'])),
            'learning_rate': best['learning_rate'],
            'min_child_weight': best['min_child_weight'],
            'reg_alpha': best['reg_alpha'],  # L1 regularization
            'reg_lambda': best['reg_lambda'],  # L2 regularization
            'tree_method': 'gpu_hist',  # Use GPU
            'predictor': 'gpu_predictor',  # Use GPU for predictions
            'sampling_method': 'gradient_based',
            'eval_metric': 'rmse'}
    best_model = xgb.train(params, xgb.DMatrix(X_train,label=y_train), rounds1)
    
    y_pred = best_model.predict(xgb.DMatrix(X_val))
    val_loss = mean_squared_error(y_val, y_pred)
    val_r2 =  r2_score(y_val, y_pred)
    val_losses.append(val_loss)
    val_r2s.append(val_r2)
    print(f"Validation loss: {val_r2} for {file}")
    print(f"Time taken: {time.time()-start}")
    xgb_models.append((model_num,centre,best_model, val_loss, val_r2))
    



#save the models
np.save(os.getcwd()+f"/cluster_model/batches/b{batch}/xgb_models.npy", xgb_models, allow_pickle=True)

val_r2s = np.array(val_r2s)
val_losses = np.array(val_losses)

#find the number of None values
none_count = np.sum(val_r2s == 0)
print(f"Number of models with no data: {none_count}")

# #plot the validation r2 scores
# plt.plot(val_r2s)
# plt.xlabel("Model number")
# plt.ylabel("Validation R2")
# plt.title(f"Validation R2 scores for batch {batch}")
# plt.savefig(os.getcwd()+f"/cluster_model/batches/b{batch}/val_r2_scores.png")
    
# #plot histogram of validation r2 scores
# plt.hist(val_r2s)
# plt.xlabel("Validation R2")
# plt.ylabel("Frequency")
# plt.title(f"Validation R2 scores for batch {batch}")
# plt.savefig(os.getcwd()+f"/cluster_model/batches/b{batch}/val_r2_hist.png")