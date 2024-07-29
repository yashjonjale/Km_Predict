import sys
# !{sys.executable} -m pip install --user -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de typing-extensions==3.7.4 --upgrade xgboost
# !{sys.executable} -m pip install --user -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de typing-extensions==3.7.4 --upgrade biopython

# !{sys.executable} -m pip install --user -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de typing-extensions==3.7.4 --upgrade hyperopt
    
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, rand
import xgboost as xgb
import numpy as np
import os
from sklearn.metrics import r2_score
import logging

#setup looger
import logging.handlers

# extract cmd arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--size', type=float, help=' percent split size')
args = parser.parse_args()
sz = args.size


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
log_file_path = f"./logs/model_4_{sz}.log"
logger = setup_logger(log_file_path)
logger.info("Starting model_4.py")



# path = "/home/not81yan/km_predict_proj/Model/Dataset"
path = os.getcwd()+"/Dataset/final_data.npz"
arr = np.load(path)
arr = arr['arr_0']

# test_arr = arr[:1000,:]
X = arr[:, 3:]
y = arr[:, 2]
real = arr[:, 1]#km_wild
X_train, X_test, y_train, y_test,r_train,r_test = train_test_split(X, y,real, test_size=(1-sz), random_state=42)
D_train=xgb.DMatrix(data=X_test, label=y_test)
D_test=xgb.DMatrix(data=X_train, label=y_train)

print("Data loaded successfully")
logger.info("Data loaded successfully")

# #now do k-fold cross validation split of X_train, y_train   
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# splits_test = []
# splits_train = []

# for train_index, test_index in kf.split(X_train):
#     splits_test.append(test_index)
#     splits_train.append(train_index)
    
# print("Data split into 5 folds")
# logger.info("Data split into 5 folds")    

R2s = []

def cross_validation_mse_gradient_boosting(params):
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
    splits_test = []
    splits_train = []

    for train_index, test_index in kf.split(X_train):
        splits_test.append(test_index)
        splits_train.append(train_index)
    for i in range(5):
        train_index, test_index  = splits_train[i], splits_test[i]
        dtrain = xgb.DMatrix(X_train[train_index], label = y_train[train_index])
        dvalid = xgb.DMatrix(X_train[test_index])
        bst = xgb.train(params, dtrain, int(num_round), verbose_eval=False)
        y_valid_pred = bst.predict(dvalid)
        MSE.append(np.mean(abs(np.reshape(y_train[test_index], (-1)) - y_valid_pred)**2))
        R2.append(r2_score(y_train[test_index] + r_train[test_index] ,  y_valid_pred+r_train[test_index]))
    R2s.append(np.median(R2))
    return(-np.median(R2))




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
    "num_rounds":  hp.uniform("num_rounds", 20, 200)
}

print("Starting hyperparameter optimization")
logger.info("Starting hyperparameter optimization")
trials = Trials()
best = fmin(fn = cross_validation_mse_gradient_boosting, space = space,
            algo=rand.suggest, max_evals = 250, trials=trials)

#date 
import datetime
now = datetime.datetime.now()

# best = None
# dump the best parameters to a file json
import json
with open(f"./logs/best_params_{sz}_size_{now}.json", 'w') as f:
    f.write(json.dumps(best))
    
# # open the file and read the best parameters
# with open('best_params_3200.json', 'r') as f:
#     best = json.load(f)
    
# best = {'learning_rate': 0.0857841089663601, 'max_delta_step': 4.510711872604566, 'max_depth': 5.0, 'min_child_weight': 8.088402920307914, 'num_rounds': 145.28674517542885, 'reg_alpha': 3.613466780469876, 'reg_lambda': 4.528913639634133}

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
best_model = xgb.train(params, xgb.DMatrix(X_train,label=y_train), int(num_round), verbose_eval=False)


# Evaluate the final model on the test set
# test_accuracy = best_model.score(X_test, y_test)
predictions = best_model.predict(xgb.DMatrix(X_test, label=y_test)) 
test_accuracy = r2_score(y_test, predictions)
print("Test Accuracy: {:.2f}".format(test_accuracy))
logger.info("Test Accuracy: {:.2f}".format(test_accuracy))


# scatter plot of the actual vs predicted values
# import matplotlib.pyplot as plt
# plt.scatter(y_test, predictions)
# plt.xlabel('Actual values')
# plt.ylabel('Predicted values')
# plt.title('Actual vs Predicted values')
# plt.show()


# apply the model on the subset of the data

# 1 - subset of test set where the wild type in the pair is not present in the training set
# 2 - subset of test set where the wild type in the pair is present 1-5 times in the training set
# etc.
# plot a histogram for that

# 3 - scatter plots and direction accuracy for different orders of magnitude


# substrate fingerprint addition

## addition of sabio rk database

# plot r2 score vs number of training samples