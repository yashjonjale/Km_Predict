import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import make_regression
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import seaborn as sns
import logging
import logging.handlers
import sys

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
log_file_path = f"./logs/tsne_tests.log"
logger = setup_logger(log_file_path)
logger.info("Starting classifier.py")



path = "./Dataset/final_data.npz"
data = np.load(path)


data = data['arr_0']
esm = data[:, 3:1280+3]
esm1 = data[:, 1283:1280+1283]
delta = data[:,2]
wild = data[:,1]


X = esm + esm1
y = delta + wild

# Normalize the target for better visualization
scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1)).ravel()

# Define a grid of t-SNE hyperparameters
param_grid = {
    'n_components': [2],  # You can add 3 if you want 3D plots
    'perplexity': [5, 10,20,30,40,50,60,70],
    'learning_rate': [200]
}

# Setting up the grid search
best_kl_divergence = np.inf
best_params = None
best_embedding = None

for params in ParameterGrid(param_grid):
    tsne = TSNE(**params, random_state=42)
    #save the embeddings as a csv file
    embedding = tsne.fit_transform(X)
    df_tsne = pd.DataFrame(embedding, columns=['Component 1', 'Component 2'])
    df_tsne['Label'] = y
    df_tsne.to_csv(f'./Dataset/tsne_plots/tsne_{params["perplexity"]}.csv', index=False)
    
    kl_divergence = tsne.kl_divergence_
    if kl_divergence < best_kl_divergence:
        best_kl_divergence = kl_divergence
        best_params = params
        best_embedding = embedding

# Output the best parameters and KL divergence
print("Best Parameters:", best_params)
print("Best KL Divergence:", best_kl_divergence)
logger.info(f"Best Parameters: {best_params}")
logger.info(f"Best KL Divergence: {best_kl_divergence}")

#save the best embedding as a csv file
df_tsne = pd.DataFrame(best_embedding, columns=['Component 1', 'Component 2'])
df_tsne['Label'] = y
df_tsne.to_csv(f'./Dataset/tsne_plots/tsne_best.csv', index=False)


# Visualizing the best embedding
# plt.figure(figsize=(8, 6))
# scatter = plt.scatter(best_embedding[:, 0], best_embedding[:, 1], c=y_scaled, cmap='viridis', edgecolor='k')
# plt.colorbar(scatter)
# plt.title('Best t-SNE embedding')
# plt.xlabel('t-SNE 1')
# plt.ylabel('t-SNE 2')
# plt.grid(True)
# plt.show()
