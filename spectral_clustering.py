import matplotlib.pyplot as plt
import numpy as np
import pickle

def spectral(data, labels, params_dict):
    sigma = params_dict['sigma']
    k = params_dict['k']

    # Affinity matrix using Gaussian kernel
    from scipy.spatial.distance import cdist
    distances = cdist(data, data, 'sqeuclidean')
    affinity_matrix = np.exp(-distances / (2. * sigma ** 2))

    # Laplacian matrix
    D = np.diag(affinity_matrix.sum(axis=1))
    L = D - affinity_matrix

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    # Select the k smallest eigenvectors
    X = eigenvectors[:, :k]

    # Normalize rows to unit length
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    # K-means clustering
    from scipy.cluster.vq import kmeans2
    centroids, computed_labels = kmeans2(X, k, minit='points')

    # SSE calculation in the reduced space
    clusters = [X[computed_labels == i] for i in range(k)]
    SSE = sum(np.sum((cluster - centroids[i]) ** 2) for i, cluster in enumerate(clusters))

    # Simplified ARI calculation
    random_index = 1 - np.sum((labels != computed_labels).astype(int)) / len(labels)

    return computed_labels, SSE, random_index, eigenvalues

def spectral_clustering():
    answers = {}
    data = np.load('cluster_data.npy')
    labels = np.load('cluster_labels.npy')
    params_dict = {'sigma': 1.0, 'k': 5}
    computed_labels, SSE, ARI, eigenvalues = spectral(data[:10000], labels[:10000], params_dict)
    
    answers["spectral_function"] = spectral
    answers["computed_labels"] = computed_labels
    answers["SSE"] = SSE
    answers["ARI"] = ARI
    
    plt.scatter(data[:10000, 0], data[:10000, 1], c=computed_labels)
    plt.title('Spectral Clustering Results')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)
    plt.show()

    return answers

if __name__ == "__main__":
    all_answers = spectral_clustering()
    with open("spectral_clustering.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)
