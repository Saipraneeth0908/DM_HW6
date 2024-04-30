import matplotlib.pyplot as plt
import numpy as np
import pickle

def gaussian_kernel(distance, sigma):
    return np.exp(-distance**2 / (2 * sigma**2))

def density_attractor(x, data, sigma, xi, max_iter=100):
    point = x
    for i in range(max_iter):
        distances = np.linalg.norm(data - point, axis=1)
        weights = gaussian_kernel(distances, sigma)
        new_point = np.sum(data * weights[:, None], axis=0) / np.sum(weights)
        if np.linalg.norm(new_point - point) < xi:
            break
        point = new_point
    return point

def denclue(data, labels, params_dict):
    sigma = params_dict['sigma']
    xi = params_dict['xi']
    # Finding density attractors
    attractors = np.array([density_attractor(x, data, sigma, xi) for x in data])
    # Cluster assignments
    clusters = {tuple(attractor): [] for attractor in attractors}
    for i, attractor in enumerate(attractors):
        clusters[tuple(attractor)].append(i)
    computed_labels = np.zeros(data.shape[0], dtype=np.int32)
    for i, cluster in enumerate(clusters.values()):
        for idx in cluster:
            computed_labels[idx] = i

    # SSE calculation
    SSE = np.sum([np.sum(np.linalg.norm(data[clusters[tuple(attractor)]]
                        - attractor, axis=1)**2) for attractor in clusters])

    # ARI calculation (simplified version)
    random_index = 1 - np.sum((labels != computed_labels).astype(int)) / len(labels) # Inverted simple matching coefficient

    return computed_labels, SSE, random_index

def denclue_clustering():
    answers = {}
    data = np.load('cluster_data.npy')
    labels = np.load('cluster_labels.npy')
    params_dict = {'sigma': 1.0, 'xi': 0.1}
    computed_labels, SSE, ARI = denclue(data[:10000], labels[:10000], params_dict)
    
    answers["denclue_function"] = denclue
    answers["computed_labels"] = computed_labels
    answers["SSE"] = SSE
    answers["ARI"] = ARI
    
    if computed_labels is not None:
        plt.scatter(data[:10000, 0], data[:10000, 1], c=computed_labels)
        plt.title('DENCLUE Clustering Results')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.grid(True)
        plt.show()
    else:
        print("No computed labels available")
    return answers

if __name__ == "__main__":
    all_answers = denclue_clustering()
    with open("denclue_clustering.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)
