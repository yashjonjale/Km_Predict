import sys    
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, rand
import xgboost as xgb
import numpy as np
import os
from sklearn.metrics import r2_score
import logging
from hyperopt import hp, STATUS_OK, Trials, fmin, tpe
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import logging.handlers

# extract cmd arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--evals', type=int, help=' num evals')
parser.add_argument('--sz', type=float, help=' test size')
args = parser.parse_args()
sz = args.sz
evals = args.evals

np.random.seed(42)

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
log_file_path = f"./logs/classifer_{evals}_{sz}.log"
logger = setup_logger(log_file_path)
logger.info("Starting classifier.py")

path = os.getcwd()+"/Dataset/data_binary.npz"
arr = np.load(path)
arr = arr['arr_0']

# shuffle the data

np.random.shuffle(arr)

X = arr[:, 3:]
y = arr[:, 2]
real = arr[:, 1]#km_wild
X_train, X_test, y_train, y_test,r_train,r_test = train_test_split(X, y,real, test_size=(1-sz), random_state=42)
D_train=xgb.DMatrix(data=X_test, label=y_test)
D_test=xgb.DMatrix(data=X_train, label=y_train)

print("Data loaded successfully")
logger.info("Data loaded successfully")

#now do k-fold cross validation split of X_train, y_train   
kf = KFold(n_splits=5, shuffle=True, random_state=42)

rocs = []
acc_s = []

rocs_train = []
acc_s_train = []

def objective(space):
    rnds = int(space['rounds'])   
    auc_roc = []
    accuracy = []    
    splits_test = []
    splits_train = []
    accuracy_train = []
    auc_roc_train = []
    for train_index, test_index in kf.split(X_train):
        splits_test.append(test_index)
        splits_train.append(train_index)
    for i in range(5):
        train_index, test_index  = splits_train[i], splits_test[i]
        dtrain = xgb.DMatrix(X_train[train_index], label = y_train[train_index])
        dvalid = xgb.DMatrix(X_train[test_index])
        eval_set  = [(X_train[train_index], y_train[train_index]), (X_train[test_index], y_train[test_index])]
        clf = XGBClassifier(
            objective='binary:logistic',
            n_estimators=int(space['n_estimators']),
            max_depth=int(space['max_depth']),
            gamma=space['gamma'],
            reg_alpha=int(space['reg_alpha']),
            reg_lambda=int(space['reg_lambda']),
            min_child_weight=space['min_child_weight'],
            colsample_bytree=space['colsample_bytree'],
            learning_rate=space['learning_rate'],
            tree_method='gpu_hist',  # Use GPU
            predictor='gpu_predictor'  # Optional: Use GPU for prediction
        )

        clf.fit(X_train[train_index], y_train[train_index],eval_set=eval_set, eval_metric="auc",early_stopping_rounds=rnds)
        y_valid_pred = clf.predict(X_train[test_index])
        accuracy.append(accuracy_score(y_train[test_index], y_valid_pred))
        auc_roc.append(roc_auc_score(y_train[test_index], y_valid_pred))
        y_train_pred = clf.predict(X_train[train_index])
        accuracy_train.append(accuracy_score(y_train[train_index], y_train_pred))
        auc_roc_train.append(roc_auc_score(y_train[train_index], y_train_pred))
    acc_s.append(np.median(accuracy))
    rocs.append(np.median(auc_roc))
    rocs_train.append(np.median(auc_roc_train))
    acc_s_train.append(np.median(accuracy_train))
    print(f"The scores for the current evaluation are:")
    print(f"Validation -")
    logger.info(f"The scores for the current evaluation are:")
    logger.info(f"Validation -")
    print("ACC: ", np.median(accuracy))
    print("ROC: ", np.median(auc_roc))
    logger.info(f"ACC: {np.median(accuracy)}")
    logger.info(f"ROC {np.median(auc_roc)}")
    print(f"Train -")
    logger.info(f"Train -")
    print(f"ACC: {np.median(accuracy_train)}")
    print(f"ROC: {np.median(auc_roc_train)}")
    logger.info(f"ACC: {np.median(accuracy_train)}")
    logger.info(f"ROC: {np.median(auc_roc_train)}")    
    return {'loss': -np.median(accuracy), 'status': STATUS_OK }


space = {
    'max_depth': hp.quniform("max_depth", 4, 10, 1),
    'gamma': hp.uniform ('gamma', 0,5),
    'reg_alpha' : hp.uniform('reg_alpha', 0,10),
    'reg_lambda' : hp.uniform('reg_lambda', 0,1),
    'colsample_bytree' : hp.uniform('colsample_bytree', 0.5, 1.5),
    'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
    'n_estimators': hp.quniform('n_estimators', 100, 1000, 1),
    'learning_rate': hp.quniform('learning_rate', 0.01, 0.1, 0.001),
    'rounds': hp.quniform('rounds', 50, 100, 10)
}


print("Starting hyperparameter optimization")
logger.info("Starting hyperparameter optimization")
trials = Trials()
best = fmin(fn = objective, space = space, algo=rand.suggest, max_evals = int(evals), trials=trials)

#date 
import datetime
now = datetime.datetime.now()

import json
with open(f"./logs/best_params_classification_{evals}_{sz}_size_{now}.json", 'w') as f:
    f.write(json.dumps(best))
    
print("Best parameters found by Hyperopt:")
print(best)
logger.info("Best parameters found by Hyperopt:")
logger.info(f"{best}")




rnds = best['rounds']
clf1 = XGBClassifier(
    objective='binary:logistic',
    n_estimators=int(best['n_estimators']),
    max_depth=int(best['max_depth']),
    gamma=best['gamma'],
    reg_alpha=int(best['reg_alpha']),
    reg_lambda=int(best['reg_lambda']),
    min_child_weight=best['min_child_weight'],
    colsample_bytree=int(best['colsample_bytree']),
    learning_rate=best['learning_rate']
)
eval_set  = [(X_train, y_train), (X_test, y_test)]
clf1.fit(X_train, y_train,eval_set=eval_set, eval_metric="auc",early_stopping_rounds=rnds, verbose=False)

pred = clf1.predict(X_test)
test_accuracy = accuracy_score(y_test, pred)


# Evaluate the final model on the test set
# test_accuracy = best_model.score(X_test, y_test)

#print evals0
# print("Evals: ", evals, "Size: ", sz)
logger.info(f"Evals: {evals} Size: {sz}")
print("Test Accuracy: {:.2f}".format(test_accuracy))
logger.info("Test Accuracy: {:.2f}".format(test_accuracy))

# measure the train accuracy
train_predictions = clf1.predict(X_train)
train_accuracy = accuracy_score(y_train, train_predictions)
print("Train Accuracy: {:.2f}".format(train_accuracy))
logger.info("Train Accuracy: {:.2f}".format(train_accuracy))
# measure validation accuracy
validation_accuracy = acc_s[-1]
print("Validation Accuracy: {:.2f}".format(validation_accuracy))    
print("ROC: ", rocs[-1])
logger.info("Validation Accuracy: {:.2f}".format(validation_accuracy))
logger.info("ROC: {:.2f}".format(rocs[-1]))


#save the list of accuracies as npy file
np.save(f"./logs/accs_{evals}_{sz}.npy", acc_s)
np.save(f"./logs/rocs_{evals}_{sz}.npy", rocs)
np.save(f"./logs/accs_train_{evals}_{sz}.npy", acc_s_train)
np.save(f"./logs/rocs_train_{evals}_{sz}.npy", rocs_train)

# save the model
clf1.save_model(f"./logs/model_classf_{evals}_{sz}.json")


# load the model from the saved json file

# loaded_model = xgb.Booster()
# loaded_model.load_model(f"./logs/model_{evals}_{sz}.json")

# # make predictions
# predictions = loaded_model.predict(D_test)
