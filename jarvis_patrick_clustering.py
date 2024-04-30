import numpy as np
import matplotlib.pyplot as plt
import pickle

def compute_neighbors(data, k):
    n = data.shape[0]
    distances = np.linalg.norm(data[:, np.newaxis] - data[np.newaxis, :], axis=2)
    neighbors = np.argsort(distances, axis=1)[:, 1:k+1]  # Exclude self
    return neighbors

def jarvis_patrick(data, labels, params_dict):
    k = params_dict['k']
    smin = params_dict['smin']
    neighbors = compute_neighbors(data, k)
    
    cluster_id = 0
    cluster_labels = -np.ones(data.shape[0], dtype=int)  # Initialize cluster labels as -1 (unassigned)

    for i in range(data.shape[0]):
        if cluster_labels[i] == -1:  # If not yet assigned
            cluster_labels[i] = cluster_id
            for j in range(data.shape[0]):
                if i != j:
                    # Intersection of neighbors
                    shared_neighbors = len(set(neighbors[i]) & set(neighbors[j]))
                    if shared_neighbors >= smin and (cluster_labels[j] == -1 or cluster_labels[j] == cluster_labels[i]):
                        cluster_labels[j] = cluster_id
            cluster_id += 1
    
    # Calculate SSE
    clusters = {new_label: data[cluster_labels == new_label] for new_label in range(cluster_id)}
    SSE = sum(np.sum((cluster - cluster.mean(axis=0)) ** 2) for cluster in clusters.values())

    # Simplified ARI
    random_index = 1 - np.sum((labels != cluster_labels).astype(int)) / len(labels)

    return cluster_labels, SSE, random_index

def jarvis_patrick_clustering():
    answers = {}
    data = np.load('cluster_data.npy')
    labels = np.load('cluster_labels.npy')
    params_dict = {'k': 5, 'smin': 4}
    computed_labels, SSE, ARI = jarvis_patrick(data[:10000], labels[:10000], params_dict)

    answers["jarvis_patrick_function"] = jarvis_patrick
    answers["computed_labels"] = computed_labels
    answers["SSE"] = SSE
    answers["ARI"] = ARI
    
    plt.scatter(data[:10000, 0], data[:10000, 1], c=computed_labels)
    plt.title('Jarvis-Patrick Clustering Results')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)
    plt.show()

    return answers

if __name__ == "__main__":
    all_answers = jarvis_patrick_clustering()
    with open("jarvis_patrick_clustering.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)
