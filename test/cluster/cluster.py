import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import networkx as nx
import ast
import plotly.graph_objects as go
from multiprocessing import Pool


class KMeansPyTorch:
    def __init__(self, num_clusters, num_iterations=100, batch_size=1000, device='cuda'):
        self.num_clusters = num_clusters
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.cluster_centers = None
        self.device = device

    def fit(self, data):
        num_samples, num_features = data.shape
        data_tensor = torch.tensor(data, dtype=torch.float32).to(self.device)
        initial_indices = np.random.choice(num_samples, self.num_clusters, replace=False)
        self.cluster_centers = data_tensor[initial_indices].clone()

        def compute_distance(x, centers):
            return torch.cdist(x, centers, p=2)  # Euclidean distance

        for iteration in range(self.num_iterations):
            print(f"Iteration {iteration + 1}/{self.num_iterations}")
            all_assignments = []
            for batch in DataLoader(TensorDataset(data_tensor), batch_size=self.batch_size, shuffle=False):
                distances = compute_distance(batch[0], self.cluster_centers)
                assignments = torch.argmin(distances, dim=1)
                all_assignments.append(assignments)
            all_assignments = torch.cat(all_assignments)
            new_centers = []
            for cluster_idx in range(self.num_clusters):
                cluster_points = data_tensor[all_assignments == cluster_idx]
                if len(cluster_points) > 0:
                    new_center = cluster_points.mean(dim=0)
                else:
                    new_center = self.cluster_centers[cluster_idx]  # Keep the same if no points are assigned
                new_centers.append(new_center)
            self.cluster_centers = torch.stack(new_centers)

    def predict(self, data):
        data_tensor = torch.tensor(data, dtype=torch.float32).to(self.device)
        def compute_distance(x, centers):
            return torch.cdist(x, centers, p=2)  # Euclidean distance
        final_assignments = []
        for batch in DataLoader(TensorDataset(data_tensor), batch_size=self.batch_size, shuffle=False):
            distances = compute_distance(batch[0], self.cluster_centers)
            assignments = torch.argmin(distances, dim=1)
            final_assignments.append(assignments)
        final_assignments = torch.cat(final_assignments)
        return final_assignments.cpu().numpy()

    def get_cluster_centers(self):
        return self.cluster_centers.cpu()

    def compute_inertia(self, data):
        data_tensor = torch.tensor(data, dtype=torch.float32).to(self.device)
        def compute_distance(x, centers):
            return torch.cdist(x, centers, p=2)  # Euclidean distance
        inertia = 0.0
        for batch in DataLoader(TensorDataset(data_tensor), batch_size=self.batch_size, shuffle=False):
            distances = compute_distance(batch[0], self.cluster_centers)
            min_distances = torch.min(distances, dim=1).values
            inertia += torch.sum(min_distances**2).item()
        return inertia

    def visualize_clusters(self, data, assignments):
        pca = PCA(n_components=2)
        data_reduced = pca.fit_transform(data)
        plt.figure(figsize=(10, 6))
        for i in range(self.num_clusters):
            cluster_points = data_reduced[assignments == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}')
        plt.legend()
        plt.title('Cluster Assignments Visualization')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.grid(True)
        plt.show()

    def visualize_cluster_graph(self, data, assignments, labels, output_file="cluster_graph.html"):
        G = nx.Graph()
        cluster_centers = self.get_cluster_centers().numpy()
        assignments = np.array(assignments)
        labels = np.array(labels)
        cluster_sizes = [(assignments == i).sum().item() for i in range(self.num_clusters)]
        cluster_labels = []
        for i in range(self.num_clusters):
            cluster_label_counts = np.bincount(labels[assignments == i])
            most_common_label = cluster_label_counts.argmax()
            most_common_label_count = cluster_label_counts[most_common_label]
            percentage = (most_common_label_count / cluster_sizes[i]) * 100
            cluster_labels.append((most_common_label, percentage))
        for i, (center, (label, percentage)) in enumerate(zip(cluster_centers, cluster_labels)):
            G.add_node(i, size=cluster_sizes[i], title=f'Size: {cluster_sizes[i]}\nLabel: {label} ({percentage:.2f}%)')
        for i in range(self.num_clusters):
            for j in range(i + 1, self.num_clusters):
                distance = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
                G.add_edge(i, j, weight=float(distance))

        pos = nx.spring_layout(G)

        # Create edge trace
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        # Create node trace
        node_x = []
        node_y = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                ),
            )
        )

        node_adjacencies = []
        node_text = []
        for node, adjacencies in enumerate(G.adjacency()):
            node_adjacencies.append(len(adjacencies[1]))
            node_info = G.nodes[node]['title']
            node_text.append(node_info)

        node_trace.marker.color = node_adjacencies
        node_trace.text = node_text

        fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                        title='<br>Network graph visualization',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper"
                        )],
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False))
                        )

        fig.write_html(os.getcwd()+"/cluster/"+output_file)
        print(f"Graph saved to {output_file}")

def kmeans_inertia(num_clusters, data, num_iterations, batch_size, device):
    kmeans = KMeansPyTorch(num_clusters=num_clusters, num_iterations=num_iterations, batch_size=batch_size, device=device)
    kmeans.fit(data)
    inertia = kmeans.compute_inertia(data)
    return inertia

def plot_elbow_curve(data, max_clusters, num_iterations=100, batch_size=1000, num_processes=4, device='cuda'):
    with Pool(num_processes) as pool:
        print("grt")
        tasks = [(k, data, num_iterations, batch_size, device) for k in range(1, max_clusters + 1)]
        inertias = pool.starmap(kmeans_inertia, tasks)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), inertias, marker='o')
    plt.title('Elbow Curve')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.show()


import os
path =  os.getcwd()+"/brenda_analyse/total_esm.csv"
import pandas as pd
df = pd.read_csv(path)
ec = df['EC_ID'].tolist()
y = np.array(df['num_value_gm'].tolist())
df['esm'] = df['esm'].apply(ast.literal_eval)
X = np.array(df['esm'].tolist())
labels = [int(x.split('.')[0]) for x in ec]
data = X.copy()

# optimal_clusters = 50  # Reduced number of clusters
# kmeans = KMeansPyTorch(num_clusters=optimal_clusters, num_iterations=100, batch_size=1000, device='cuda')
# kmeans.fit(data)

# assignments = kmeans.predict(data)

# kmeans.visualize_cluster_graph(data, assignments, labels, output_file="cluster_graph.html")

sizes = [10,20,50,100,200,300,400,500,600,700,800,900,1000]
centres = []

for size in sizes:
    kmeans = KMeansPyTorch(num_clusters=size, num_iterations=170, batch_size=1000, device='cuda')
    kmeans.fit(data)
    assignments = kmeans.predict(data)
    kmeans.visualize_cluster_graph(data, assignments, labels, output_file=f"cluster_graph_{size}.html")
    print(f"Graph saved to cluster_graph_{size}.html")
    #store the cluster centres
    centres.append(kmeans.get_cluster_centers().numpy())
    print(f"Cluster centers for {size} clusters saved")
    
#save the cluster centres as a numpy array
np.save(os.getcwd()+"/cluster/cluster_centres.npy", np.array(centres))


