import numpy as np
import pandas as pd
import logging
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ParameterGrid
import umap
import argparse

# extract cmd arguments
parser = argparse.ArgumentParser()
parser.add_argument('--evals', type=int, help=' num evals')
args = parser.parse_args()
evals = args.evals


def setup_logger(log_file_path):
    # Create a logger
    logger = logging.getLogger('umap_logger')
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
log_file_path = f"./logs/umap_tests.log"
logger = setup_logger(log_file_path)
logger.info("Starting UMAP script")

# Load the dataset
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

Num = evals
#Do PCA to convert 1280 features to Num parameters
from sklearn.decomposition import PCA
pca = PCA(n_components=Num)
X = pca.fit_transform(X)



# Define a grid of UMAP hyperparameters
param_grid = {
    'n_neighbors': [5, 10, 15, 20, 25, 30],
    'min_dist': [0.1, 0.2, 0.3, 0.4, 0.5],
    'n_components': [2]  # You can add 3 if you want 3D plots
}

# Setting up the grid search
best_loss = np.inf
best_params = None
best_embedding = None

for params in ParameterGrid(param_grid):
    umap_model = umap.UMAP(**params, random_state=42)
    # Save the embeddings as a csv file
    embedding = umap_model.fit_transform(X)
    df_umap = pd.DataFrame(embedding, columns=['Component 1', 'Component 2'])
    df_umap['Label'] = y
    df_umap.to_csv(f'./Dataset/umap_plots/umap_{params["n_neighbors"]}_{params["min_dist"]}.csv', index=False)

    # UMAP does not have KL divergence, so we'll use reconstruction error
    reconstruction_error = np.mean((X - umap_model.inverse_transform(embedding))**2)
    if reconstruction_error < best_loss:
        best_loss = reconstruction_error
        best_params = params
        best_embedding = embedding

# Output the best parameters and reconstruction error
print("Best Parameters:", best_params)
print("Best Reconstruction Error:", best_loss)
logger.info(f"Best Parameters: {best_params}")
logger.info(f"Best Reconstruction Error: {best_loss}")

# Save the best embedding as a csv file
df_umap = pd.DataFrame(best_embedding, columns=['Component 1', 'Component 2'])
df_umap['Label'] = y
df_umap.to_csv(f'./Dataset/umap_plots/umap_best.csv', index=False)

# Visualizing the best embedding
plt.figure(figsize=(8, 6))
scatter = plt.scatter(best_embedding[:, 0], best_embedding[:, 1], c=y_scaled, cmap='viridis', edgecolor='k')
plt.colorbar(scatter)
plt.title('Best UMAP embedding')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.grid(True)
plt.show()
