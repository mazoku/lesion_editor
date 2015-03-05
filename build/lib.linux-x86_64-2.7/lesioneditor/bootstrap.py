__author__ = 'tomas'

from sklearn.cluster import KMeans
import numpy as np


def generate_bootstrap_pairs(data, B, n_observations=1):
    n = data.size
    bpairs_idxs = np.random.randint(0, n, (2, np.round(n_observations * n).astype(np.int), B))
    bpairs_dat = data[np.unravel_index(bpairs_idxs.flatten(), data.shape)].reshape(bpairs_idxs.shape)

    # for i in range(bpairs.shape[2]):
    #     print bpairs_idxs[:,:,i]

    return bpairs_idxs, bpairs_dat


def classify(bpairs_dat, k):
    B = bpairs_dat.shape[2]
    kmeans = KMeans(k)
    kmeans.fit(bpairs_dat)
    labels = kmeans.labels_

    return labels


def calculate_clustering_distance():



def run(data, B, n_observations, k_min, k_max):
    n_k = k_max - k_min + 1
    instabs = np.zeros()
    for k in range(k_min, k_max + 1):
        # step 1 - generating B bootstrap sample-pairs
        bpairs_idxs, bpairs_dat = generate_bootstrap_pairs(data, B, n_observations)

        # step 2 - determine the total number of (n * 2) clusterings
        # labels = classify(bpairs_dat, k)
        clusterings = classify(bpairs_dat, k)

        # step 3 - calculate empirical clustering distance
        calculate_clustering_distance()


if __name__ == '__main__':
    k_min = 2
    k_max = 5
    n = 10
    B = 5
    n_observations = 1
    run(data, B, n_observations, k_min, k_max)