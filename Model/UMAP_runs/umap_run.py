import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
import optuna
import os
import ast

#use argparse to get the number of components
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--components', type=int, default=50, help='Number of components to reduce the data to')
args = parser.parse_args()
cmps = args.components


# Load your dataset (replace with your actual dataset)
# Assuming the dataset is a CSV file with features and a continuous target column named 'target'
pth = os.getcwd()+"/../Dataset/brenda_analyse/total_esm.csv"
data = pd.read_csv(pth)
esm = data['esm']
val = data['num_value_gm']
#ensure val is a float
val = val.astype(float)
esm = esm.apply(ast.literal_eval)
y = val.to_numpy()
X = np.array(esm.to_list()) 

# what is the expected shape of X and y? according to the below code, X should be a 2D array and y should be a 1D array
# If X is a 2D array, then the shape of X should be (n_samples, n_features) and the shape of y should be (n_samples,)

# Normalize the data
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Dimensionality reduction using PCA
pca = PCA(n_components=cmps)  # Reducing to 50 components
X_reduced = pca.fit_transform(X_normalized)

# Function to optimize UMAP hyperparameters
def objective(trial):
    n_neighbors = trial.suggest_int('n_neighbors', 2, 200)
    min_dist = trial.suggest_uniform('min_dist', 0.0, 0.99)
    n_components = trial.suggest_int('n_components', 2, 50)
    
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=42
    )
    embedding = reducer.fit_transform(X_reduced)
    
    # Using a placeholder metric for optimization, e.g., variance explained by the components
    # You can replace this with a more meaningful metric for your use case
    pca_check = PCA(n_components=n_components)
    pca_check.fit(embedding)
    variance_explained = np.sum(pca_check.explained_variance_ratio_)
    
    return variance_explained

# Optimize UMAP hyperparameters
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Best hyperparameters
best_params = study.best_params

# Final UMAP embedding with best hyperparameters
best_umap = umap.UMAP(
    n_neighbors=best_params['n_neighbors'],
    min_dist=best_params['min_dist'],
    n_components=best_params['n_components'],
    random_state=42
)
embedding = best_umap.fit_transform(X_reduced)

# Save or visualize the embedding
np.save(os.getcwd()+f'/UMAP_runs/umap_emb_bst.npy', embedding)

# If you want to visualize the result
import matplotlib.pyplot as plt

plt.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap='Spectral', s=5)
plt.colorbar()
plt.title('UMAP projection of the dataset')
plt.savefig(os.getcwd()+f'/UMAP_runs/umap_projection_{cmps}.png')

#save the umap parameters
with open(os.getcwd()+f'/UMAP_runs/umap_params_{cmps}.txt', 'w') as f:
    f.write(str(best_params))
