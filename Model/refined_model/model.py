import sys    
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, rand
import xgboost as xgb
import numpy as np
import os
from sklearn.metrics import r2_score
import ast


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--evals', type=float, help=' num evals')
args = parser.parse_args()
evals = args.evals



import logging
import logging.handlers

def setup_logger(log_file_path):
    # Create a logger
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)  # Set the logging level

    # Create a file handler to log messages to a file
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)  # Set the logging level for the file handler

    # Create a console handler to log messages to the console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # Set the logging level for the console handler

    # Create a formatter and set it for both handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# start the logger
log_file_path = f"./refined_model/logs/model_filtered_{evals}.log"
logger = setup_logger(log_file_path)
logger.info("Starting model.py")

def prepare_data(path):
    import ast
    df = pd.read_csv(path)
#     print(df.columns)
    print(f'The shape of the train pairs is {df.shape}')
    print(df.dtypes)
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
    sim = np.array(df['SIM'].tolist())
    #create a histogram of the SIM column and save it
    try:
        #save the sim array as a npy file
        np.save(os.getcwd()+f"/refined_model/data/sim.npy", sim)
        from matplotlib import pyplot as plt
        plt.hist(sim, bins=25)
        plt.xlabel('Similarity')
        plt.ylabel('Frequency')
        plt.title('Histogram of Similarity')
        plt.savefig(os.getcwd()+f"refined_model/data/histogram_sims.png")
    except:
        print("Could not save the histogram")
    df = df.loc[df['SIM'] >= 0.2]
    df['SEQ_REF_ID'] = df['SEQ_REF_ID'].astype(int)
    df['SEQ_MUT_ID'] = df['SEQ_MUT_ID'].astype(int)
    #drop the SIM column
    df = df.drop(columns=['SIM'])
    df_ww = df.loc[(df['CHANGE_REF'] == '-') & (df['CHANGE_MUT'] == '-')].copy()
    df_ww.reset_index(drop=True, inplace=True)
    uniq_seq_ids = set()
    for i in range(len(df_ww)):
        uniq_seq_ids.add(df_ww.loc[i, 'SEQ_REF_ID'])
        uniq_seq_ids.add(df_ww.loc[i, 'SEQ_MUT_ID'])
    seq_ids = list(uniq_seq_ids)
    return df, seq_ids

def prep_tst_data(df):
    print(f'The shape of the test pairs is {df.shape}')
    print(df.dtypes)
    df['ESM_REF'] = df['ESM_REF'].apply(ast.literal_eval)  
    df['ESM_MUT_delta'] = df['ESM_MUT_delta'].apply(ast.literal_eval)
    df['KM_REF'] = df['KM_REF'].astype(float)
    df['SEQ_REF_ID'] = df['SEQ_REF_ID'].astype(int)
    df['SEQ_MUT_ID'] = df['SEQ_MUT_ID'].astype(int)
    df['SUBSTRATE_ID'] = df['SUBSTRATE_ID'].astype(int)
    df['SIM'] = df['SIM'].astype(float)
    # delete the rows where SIM is less than 0.9
    df = df.loc[df['SIM'] >= 0.3]
    df['SEQ_REF_ID'] = df['SEQ_REF_ID'].astype(int)
    df['SEQ_MUT_ID'] = df['SEQ_MUT_ID'].astype(int)
    #drop the SIM column
    df = df.drop(columns=['SIM'])
    return df


#train = pd.DataFrame(columns=['SUBSTRATE_ID','CHANGE_REF','CHANGE_MUT','SEQ_REF_ID','SEQ_MUT_ID','ESM_REF','ESM_MUT_delta', 'KM_REF', 'KM_MUT', 'LABEL'])

path = os.getcwd()+"/refined_model/data/train_pairs_idx.csv"
df_train_pairs, ref_ids = prepare_data(path)
df_test_pairs = pd.read_csv(os.getcwd()+"/refined_model/data/test_pairs_idx.csv")
train_tup = None
test_tup = None

#for both test and train pairs, delete the columns that are not needed
df_train_pairs = df_train_pairs.drop(columns=['SEQ_REF_ID', 'SEQ_MUT_ID'])
#convert the esm column to a list using ast
# df_train_pairs['ESM_REF'] = df_train_pairs['ESM_REF'].apply(lambda x: ast.literal_eval(x))
# df_train_pairs['ESM_MUT_delta'] = df_train_pairs['ESM_MUT_delta'].apply(lambda x: ast.literal_eval(x))
esm_ref = np.array(df_train_pairs['ESM_REF'].tolist())
esm_mut = np.array(df_train_pairs['ESM_MUT_delta'].tolist())
#convert the km column to a list using ast
# df_train_pairs['KM_REF'] = df_train_pairs['KM_REF'].apply(lambda x: ast.literal_eval(x))
# df_train_pairs['KM_MUT'] = df_train_pairs['KM_MUT'].apply(lambda x: ast.literal_eval(x))
km_delta = np.array(df_train_pairs['KM_MUT'].tolist()) - np.array(df_train_pairs['KM_REF'].tolist())
km_ref = np.array(df_train_pairs['KM_REF'].tolist())
km_mut = np.array(df_train_pairs['KM_MUT'].tolist())
subs = np.array(df_train_pairs['SUBSTRATE_ID'].tolist())
#concatenate the esm columns
esm_sub = np.concatenate((esm_ref, esm_mut, subs.reshape((-1,1))), axis=1)
train_tup = (esm_sub, km_delta, km_ref, km_mut)


df_test_pairs = prep_tst_data(df_test_pairs)
#repeat the above for the test pairs
df_test_pairs = df_test_pairs.drop(columns=['SEQ_REF_ID', 'SEQ_MUT_ID'])
#convert the esm column to a list using ast
esm_ref = np.array(df_test_pairs['ESM_REF'].tolist())
esm_mut = np.array(df_test_pairs['ESM_MUT_delta'].tolist())
subs = np.array(df_test_pairs['SUBSTRATE_ID'].tolist())
#convert the km column to a list using ast
# df_test_pairs['KM_REF'] = df_test_pairs['KM_REF'].apply(lambda x: ast.literal_eval(x))
# df_test_pairs['KM_MUT'] = df_test_pairs['KM_MUT'].apply(lambda x: ast.literal_eval(x))
km_delta = np.array(df_test_pairs['KM_MUT'].tolist()) - np.array(df_test_pairs['KM_REF'].tolist())
km_ref = np.array(df_test_pairs['KM_REF'].tolist())
km_mut = np.array(df_test_pairs['KM_MUT'].tolist())
#concatenate the esm columns
esm_sub = np.concatenate((esm_ref, esm_mut, subs.reshape((-1,1))), axis=1)
test_tup = (esm_sub, km_delta, km_ref, km_mut)


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
    
    df_mw = df.loc[(df['CHANGE_REF'] != '-') | (df['CHANGE_MUT'] != '-')].copy()
    df_mw.reset_index(drop=True, inplace=True)
    #make 5 parts of the df_mw
    for i in range(len(df_mw)):
        folds[i%5] = folds[i%5].append(df_mw.loc[i], ignore_index=True)
        fld_nums[i%5] += 1
    
    df_ww = df.loc[(df['CHANGE_REF'] == '-') & (df['CHANGE_MUT'] == '-')].copy()
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


print("Data loaded successfully")
logger.info("Data loaded successfully")

ev = 0

R2s = []
MSEs = []
def cross_validation_mse_gradient_boosting(params):
    global ev
    print(f"Starting evaluation {ev}")
    logger.info(f"Starting evaluation {ev}")
    num_round = params['num_rounds']
    params = {
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
    MSE = []
    R2 = []    
    #shuffle the dataframe entries
    global df_train_pairs
    df_train_pairs = df_train_pairs.sample(frac=1).reset_index(drop=True)
    print("Shuffled the dataframe entries")
    print(f"Starting split at evaluation {ev}")
    flds = five_fold_split(df_train_pairs,ref_ids)
    print("Split done")
    logger.info("Split done")
    ev+=1
    for i in range(5):
        val_tup = flds[i]
        tr_tup = None
        fl_tmp_esm = []
        fl_tmp_km_delta = []
        fl_tmp_km_ref = []
        fl_tmp_km_mut = []
        
        for j in range(5):
            if j != i:
                fl_tmp_esm.append(flds[j][0])
                fl_tmp_km_delta.append(flds[j][1])
                fl_tmp_km_ref.append(flds[j][2])
                fl_tmp_km_mut.append(flds[j][3])
        fl_esm = np.concatenate(fl_tmp_esm, axis=0)
        fl_km_delta = np.concatenate(fl_tmp_km_delta, axis=0)
        fl_km_ref = np.concatenate(fl_tmp_km_ref, axis=0)
        fl_km_mut = np.concatenate(fl_tmp_km_mut, axis=0)
        tr_tup = (fl_esm, fl_km_delta, fl_km_ref, fl_km_mut)
        
        dtrain = xgb.DMatrix(tr_tup[0], label = tr_tup[1])
        dvalid = xgb.DMatrix(val_tup[0])
        bst = xgb.train(params, dtrain, int(num_round))
        y_valid_pred = bst.predict(dvalid)
        MSE.append(np.mean(abs(np.reshape(val_tup[1], (-1)) - y_valid_pred)**2))
        R2.append(r2_score(val_tup[1] ,  y_valid_pred))
    R2s.append(np.median(R2))
    MSEs.append(np.median(MSE))
    logger.info(f"R2s: {R2}")
    logger.info(f"MSEs: {MSE}")
        
    return(np.median(MSE))




space = {
    'max_depth': hp.quniform('max_depth', 3, 14, 1),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    # 'n_estimators': hp.quniform('n_estimators', 100, 500, 1),
    # 'gamma': hp.uniform('gamma', 0, 0.5),
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

print("Starting hyperparameter optimization")
logger.info("Starting hyperparameter optimization")
trials = Trials()
best = fmin(fn = cross_validation_mse_gradient_boosting, space = space,
            algo=rand.suggest, max_evals = evals, trials=trials)

#date 
import datetime
now = datetime.datetime.now()

import json
with open(f"./logs/best_params_0.8_size_{now}.json", 'w') as f:
    f.write(json.dumps(best))


print("Best parameters found by Hyperopt:")
print(best)
logger.info("Best parameters found by Hyperopt:")
logger.info(f"{best}")
num_round = best['num_rounds']
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
best_model = xgb.train(params, xgb.DMatrix(train_tup[0],label=train_tup[1]), int(num_round), verbose_eval=False)


# Evaluate the final model on the test set
# test_accuracy = best_model.score(X_test, y_test)

#print evals0
print(f"Evals: {evals}")
logger.info(f"Evals: {evals}")

predictions = best_model.predict(xgb.DMatrix(test_tup[0], label=test_tup[1])) 
test_accuracy = r2_score(test_tup[1], predictions)
print("Test Accuracy: {:.2f}".format(test_accuracy))
logger.info("Test Accuracy: {:.2f}".format(test_accuracy))

# measure the train accuracy
train_predictions = best_model.predict(xgb.DMatrix(train_tup[0], label=train_tup[1]))
train_accuracy = r2_score(train_tup[1], train_predictions)
print("Train Accuracy: {:.2f}".format(train_accuracy))
logger.info("Train Accuracy: {:.2f}".format(train_accuracy))
# measure validation accuracy
validation_accuracy = R2s[-1]
print("Validation Accuracy: {:.2f}".format(validation_accuracy))    
logger.info("Validation Accuracy: {:.2f}".format(validation_accuracy))

# scatter plot of the actual vs predicted values
import matplotlib.pyplot as plt
plt.scatter(test_tup[1], predictions)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Actual vs Predicted values')

#save the plot
plt.savefig(f"./logs/actual_vs_predicted_{now}.png")

#plot the MSEs curve
plt.plot(MSEs)
plt.xlabel('Iterations')
plt.ylabel('MSE')
plt.title('MSE curve')
plt.savefig(f"./logs/MSE_curve_{now}.png")

#plot the R2s curve
plt.plot(R2s)
plt.xlabel('Iterations')
plt.ylabel('R2')
plt.title('R2 curve')
plt.savefig(f"./logs/R2_curve_{now}.png")

# find the indices of test_tup[1] where the values is between 0 to 0.3, 0.3 to 0.6, 0.6 to 0.9, 0.9 to 1.2, 1.2 to 1.5
lst = []
for i in range(5):
    lst.append(np.where((test_tup[1] >= 0.3*i) & (test_tup[1] < 0.3*(i+1))))
R2s_subsets = []

for i in range(5):
    X = test_tup[0][lst[i]]
    y = test_tup[1][lst[i]]
    predictions = best_model.predict(xgb.DMatrix(X, label=y))
    R2s_subsets.append(r2_score(y, predictions))

#plot the R2s_subsets
plt.plot(R2s_subsets)
plt.xlabel('Subsets')
plt.ylabel('R2')
plt.title('R2s_subsets')
plt.savefig(f"./logs/R2s_subsets_{now}.png")

#find the number of entries in each subset
nums = []
for i in range(5):
    nums.append(len(lst[i][0]))

#plot the nums
plt.plot(nums)
plt.xlabel('Subsets')
plt.ylabel('Number of entries')
plt.title('Number of entries in each subset')
plt.savefig(f"./logs/nums_{now}.png")
logger.info("Model.py finished successfully")
print("Model.py finished successfully")



