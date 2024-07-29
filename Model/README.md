# Workflow

model_{n}.py trains the model and dumps the best parameter, and model_results.py trains on the best parameters, and gives the r2 score or the any particular esults you need


### test 1 

trained on 3200 data points got a r2 score of 0.55, with naive cross validation - 5 fold

{'learning_rate': 0.0857841089663601, 'max_delta_step': 4.510711872604566, 'max_depth': 5.0, 'min_child_weight': 8.088402920307914, 'num_rounds': 145.28674517542885, 'reg_alpha': 3.613466780469876, 'reg_lambda': 4.528913639634133}
2024-06-20 16:53:15,661 - my_logger - INFO - Best parameters found by Hyperopt:
2024-06-20 16:53:15,661 - my_logger - INFO - {'learning_rate': 0.0857841089663601, 'max_delta_step': 4.510711872604566, 'max_depth': 5.0, 'min_child_weight': 8.088402920307914, 'num_rounds': 145.28674517542885, 'reg_alpha': 3.613466780469876, 'reg_lambda': 4.528913639634133}
Test Accuracy: 0.55
2024-06-20 16:53:23,125 - my_logger - INFO - Test Accuracy: 0.55
Done with model.py
completed successfully


### test 2

Do test 1 but on the entire dataset 5 evals 
r2 score = 0.50


Remarks - 
found out that our r2 scores maybe faulty, cause we are checking r2 for the deltas and not for the real km_values
so into_numpy script will need the similarity vals and also the km_wilds

also, include the thresholds for including a pair


### test 3, 4, 5

Ran the model for the entire dataset for - 

- tuning and training both on deltas
- tuning on km values but training on deltas
- tuning and training on delta values both, but randomising validation 5 fold splits

Remarks
 - test accuracy -> 0.55, training accuracy -> 0.9
 - later test set filtered based on cosine similarity,
 - for threshold 0.9 -> ~0.55
 - for threshold 0.99 -> ~0.63
 - for threshold 0.999 -> ~0.71
 - Also, similar results when trained and tested both on the tested set



In CLEAN’s task, the amino acid sequences with the same EC number have a small Euclidean distance, whereas sequences with different EC numbers have a large distance. Contrastive losses were used to train the model with supervision 

https://www.science.org/doi/10.1126/science.adf2465


https://pubmed.ncbi.nlm.nih.gov/37414576/

https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-024-05699-5#ref-CR15

https://pubmed.ncbi.nlm.nih.gov/34665809/

https://www.nature.com/articles/nmeth.2340

When dealing with a dataset that includes 92,000 data points and 2,560 features, exploratory data analysis (EDA) can become quite challenging due to the sheer volume and dimensionality of the data. Here are some effective techniques and strategies to perform EDA on such a large and complex dataset:

### 1. **Sampling**
   - **Random Sampling**: Due to the large size of your dataset, consider taking a random sample for initial EDA to make the computations more manageable and faster.
   - **Stratified Sampling**: If there are subgroups within your data that are important (e.g., different categories or clusters), ensure these are represented proportionally in your sample.

### 2. **Dimensionality Reduction**
   - **Principal Component Analysis (PCA)**: Use PCA to reduce the number of features to a smaller set that captures the most variance in the data.
   - **t-Distributed Stochastic Neighbor Embedding (t-SNE)**: This technique is good for visualizing high-dimensional data in two or three dimensions. It can help in identifying clusters or groups.
   - **Linear Discriminant Analysis (LDA)**: Useful if your target variable is categorical (despite having continuous targets, you might still want to explore this for segmenting your features).

### 3. **Visualization Techniques**
   - **Heatmaps of Correlation**: Calculate correlations between features or between features and the target, and visualize these using heatmaps. This helps identify which features are most relevant.
   - **Pair Plots**: For a selected subset of features (possibly from PCA or correlation analysis), visualize scatter plots of feature pairs to understand relationships.
   - **Histograms and Box Plots**: Use these to understand the distribution and spread of your data for individual features. This is helpful for spotting outliers and distribution shapes.

### 4. **Statistical Summaries**
   - **Descriptive Statistics**: Compute mean, median, mode, range, variance, and standard deviation for your features to get a sense of the central tendencies and dispersion.
   - **Skewness and Kurtosis**: These measures can tell you about the tail behavior and peakedness of the distributions, which are crucial for deciding on data transformation methods.

### 5. **Feature Engineering and Selection**
   - **Feature Importance**: Utilize models that provide feature importance (like Random Forests) to identify which features are contributing most to the prediction of the target.
   - **Recursive Feature Elimination (RFE)**: Implement RFE to systematically remove features and determine which ones have the most significant impact on model performance.

### 6. **Advanced Analytics**
   - **Cluster Analysis**: Apply clustering techniques (like K-means, hierarchical clustering) to discover inherent groupings in the data.
   - **Anomaly Detection**: Use statistical tests or anomaly detection algorithms to identify outliers or unusual data points which could be influential or erroneous.

### Tools and Technologies
For handling such data efficiently, you may need powerful computational tools and environments:
   - **Python Libraries**: `scikit-learn` for machine learning, `pandas` for data manipulation, `matplotlib` and `seaborn` for plotting, `numpy` for numerical operations.
   - **Big Data Technologies**: If computational resources are a constraint, consider using Spark with PySpark to handle data at this scale more effectively.

### Considerations
- **Memory Management**: Be mindful of memory usage; operations on large datasets can often lead to memory errors.
- **Compute Power**: Use powerful machines or cloud-based services with adequate CPU and RAM to handle the computations.

The key to effective EDA with such datasets is to strategically reduce the complexity either by focusing on subsets or by using statistical techniques to summarize and visualize the data effectively. If you have any specific aspects of EDA or tools you need more details on, feel free to ask!



___
___
___

### check this paper out https://www.nature.com/articles/s41929-022-00798-z




### make a lstm based model, use all embedding vectors, zero-shot learning based model



Model Interpretability and Diagnostics
SHAP Values
Use SHAP values to understand which features are most important for the model’s predictions and focus on improving these.
python
Copy code
import shap

# Train your model
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Explain the model's predictions using SHAP
explainer = shap.Explainer(model)
shap_values = explainer(X_val)

# Plot SHAP values
shap.summary_plot(shap_values, X_val)


7. Model Interpretability
SHAP Values: Use SHAP to understand feature importance and interactions, which can provide insights for feature engineering.
LIME: Use Local Interpretable Model-agnostic Explanations (LIME) for understanding individual predictions.


3. Handling Data Sparsity
Imputation: Impute missing values using techniques like k-NN or matrix factorization.
Sparse Matrices: Convert your data to sparse matrix formats to better handle sparsity.
4. Semi-Supervised Learning
Self-Training: Label the unlabeled data using your current model and retrain on the augmented dataset.
Consistency Regularization: Ensure predictions are consistent with slight perturbations.


1. Feature Engineering
Interaction Features: Create interaction features between existing ones.
Polynomial Features: Generate polynomial features if the relationships are non-linear.
Domain Knowledge: Incorporate domain-specific features that might be informative.


1. Dimensionality Reduction
Reduce the feature space to a lower-dimensional space while preserving the most important information.

a. Principal Component Analysis (PCA)
PCA transforms the data to a lower-dimensional space by finding the directions (principal components) that maximize the variance.

Implementation in Python:

python
Copy code
from sklearn.decomposition import PCA

# Assume embeddings is your input data matrix
pca = PCA(n_components=0.95)  # Keep 95% of variance
X_reduced = pca.fit_transform(embeddings)
b. t-Distributed Stochastic Neighbor Embedding (t-SNE)
t-SNE is effective for visualizing high-dimensional data by reducing it to 2 or 3 dimensions.

Implementation in Python:

python
Copy code
from sklearn.manifold import TSNE

# Reduce dimensionality to 2D for visualization
tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
X_reduced = tsne.fit_transform(embeddings)
2. Feature Selection
Identify and select the most relevant features to reduce dimensionality.

a. Recursive Feature Elimination (RFE)
RFE recursively removes the least important features to build a model with the most significant ones.

Implementation in Python:

python
Copy code
from sklearn.feature_selection import RFE
from xgboost import XGBRegressor

# Assume embeddings is your input data matrix and targets are your labels
model = XGBRegressor()
rfe = RFE(model, n_features_to_select=10)
X_reduced = rfe.fit_transform(embeddings, targets)




### tSNE 

https://lvdmaaten.github.io/tsne/#:~:text=t%2DDistributed%20Stochastic%20Neighbor%20Embedding,on%20large%20real%2Dworld%20datasets.

