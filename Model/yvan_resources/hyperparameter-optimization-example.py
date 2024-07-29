import numpy as np
import pandas as pd
import os
from os.path import join
from sklearn.metrics import r2_score
import xgboost as xgb
import json

from hyperopt import fmin, tpe, rand, hp, Trials
CURRENT_DIR = os.getcwd()


### loading datatrain_df_raw_wo_mutant-mutant_cv3.pkl
#in hpc
data_train = pd.read_pickle("/gpfs/project/rousset/project_variant_v3/datasets/symetric/split_and_cv/train_df_ref_symetric.pkl")
data_test =  pd.read_pickle("/gpfs/project/rousset/project_variant_v3/datasets/symetric/split_and_cv/test_df_ref_symetric.pkl")


train_indices = list(np.load(join( "/gpfs/project/rousset/project_variant_v3/datasets/symetric/split_and_cv/CV_train_df_ref_symetric.npy"), allow_pickle = True))
test_indices = list(np.load(join(  "/gpfs/project/rousset/project_variant_v3/datasets/symetric/split_and_cv/CV_test_df_ref_symetric.npy"), allow_pickle = True))



######
train_X1 = np.array(list(data_train["ESM1b"]))
train_X2 = np.array(list(data_train["fp1"]))
train_X = []
for i in range(len(train_X1)):
    concatenated = np.concatenate((train_X1[i][0], train_X2[i]))
    train_X.append(concatenated)
train_X = np.array(train_X)
train_Y = np.array(list(data_train["delta_log_kcat"]))

#### hyperparameter optimisation
def cross_validation_mse_gradient_boosting(param):
    num_round = param["num_rounds"]
    del param["num_rounds"]
    param["max_depth"] = int(np.round(param["max_depth"]))
    param["tree_method"] = "gpu_hist"
    param["sampling_method"] = "gradient_based"
    
    MSE = []
    R2 = []
    for i in range(5):
        train_index, test_index  = train_indices[i], test_indices[i]
        dtrain = xgb.DMatrix(train_X[train_index], label = train_Y[train_index])
        dvalid = xgb.DMatrix(train_X[test_index])
        bst = xgb.train(param, dtrain, int(num_round), verbose_eval=False)
        y_valid_pred = bst.predict(dvalid)
        MSE.append(np.mean(abs(np.reshape(train_Y[test_index], (-1)) - y_valid_pred)**2))
        R2.append(r2_score(np.reshape(train_Y[test_index], (-1)),  y_valid_pred))
    return(-np.mean(R2))


space_gradient_boosting = {
    "learning_rate": hp.uniform("learning_rate", 0.01, 1),
    "max_depth": hp.uniform("max_depth", 4,12),
    #"subsample": hp.uniform("subsample", 0.7, 1),
    "reg_lambda": hp.uniform("reg_lambda", 0, 5),
    "reg_alpha": hp.uniform("reg_alpha", 0, 5),
    "max_delta_step": hp.uniform("max_delta_step", 0, 5),
    "min_child_weight": hp.uniform("min_child_weight", 0.1, 15),
    "num_rounds":  hp.uniform("num_rounds", 20, 200)
    #estimators
    #gamma
    #colsample_bytree
    #colsample_bylevel
    #colsample_bynode
    #sampling_method
    #tree_method
    #gpu_id
    #gpu_page_size
    #max_bin

        
    }

trials = Trials()
best = fmin(fn = cross_validation_mse_gradient_boosting, space = space_gradient_boosting,
            algo=rand.suggest, max_evals = 250, trials=trials)



#with open(join("data","best_param.json"), 'w', encoding='utf-8') as f:
with open(join("/gpfs/project/rousset/project_variant_v3/hyperopt/best_parameters/ref_symetric_fp2.json"), 'w', encoding='utf-8') as f:
    json.dump(best, f, ensure_ascii=False, indent=4)