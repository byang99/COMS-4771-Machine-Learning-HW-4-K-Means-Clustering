import random
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets as skl_data
from numpy import linalg as LA
from sklearn.neighbors import NearestNeighbors


# 3C Part i

def assign_data(data, cluster_centers):
    labels = list()

    # find the closest center for each data point x
    for d in data:
        # get row i of data 2d array
        x = np.asarray(d)

        # find closest center - calculate min squared L2 norm
        closest_center = cluster_centers[0]
        min_distance = (np.linalg.norm(x - closest_center, ord=2)) ** 2
        for c in cluster_centers:
            distance = (np.linalg.norm(x - c, ord=2)) ** 2

            if distance < min_distance:
                closest_center = c
                min_distance = distance

        # found closest center for x
        # clusters list holds the cluster center each data point
        # is associated with
        labels.append(closest_center)

    return labels


def find_optimal_centers(data, labels, cluster_centers):
    new_cluster_centers = list()

    # for each cluster, find the optimal center
    for center in cluster_centers:

        # find all data points that have this center
        # and add to a list
        cluster = list()
        for i in range(len(labels)):
            if np.array_equal(labels[i], center):
                # data point has this center. Add to cluster
                cluster.append(data[i])

        cluster = np.asarray(cluster)
        # optimal center should be the mean of all the data points in the cluster
        # take mean over columns - each dimension is averaged
        optimal_center = np.mean(cluster, axis=0)

        new_cluster_centers.append(optimal_center)

        old_labels = labels
        new_labels = old_labels.copy()

        # update all relevant data points to have new center
        for i in range(len(old_labels)):
            if np.array_equal(old_labels[i], center):
                # data point has this center. Add to cluster
                new_labels[i] = optimal_center

    return new_cluster_centers, new_labels, old_labels


def converged(cluster_centers, new_cluster_centers, new_labels, old_labels):
    if np.array_equal(cluster_centers, new_cluster_centers) and np.array_equal(new_labels, old_labels):
        return True
    else:
        return False


def k_means(data, k):
    # Input:
    # k - number of intended groupings
    # data - list of data x_1, x_2, ...,x_n in R^d

    # Begin alternating optimization

    # initialize cluster centers
    # pick k random data points from our data points
    cluster_centers = random.sample(data.tolist(), k)

    # assign data to its closest center
    labels = assign_data(data, cluster_centers)

    # find optimal centers
    new_centers, new_labels, old_labels = find_optimal_centers(data, labels, cluster_centers)

    while not converged(cluster_centers, new_centers, new_labels, old_labels):
        # assign data to its closest center
        cluster_centers = new_centers
        labels = assign_data(data, cluster_centers)

        # find optimal centers
        new_centers, new_labels, old_labels = find_optimal_centers(data, labels, cluster_centers)

    for i in range(len(cluster_centers)):
        for j in range(len(labels)):
            if np.array_equal(cluster_centers[i], labels[j]):
                labels[j] = i

    return cluster_centers, labels


def make_rectangles():
    inner_top = [[i, 2.0] for i in np.linspace(-2.0, 2.0, num=125)]
    inner_bot = [[i, -2.0] for i in np.linspace(-2.0, 2.0, num=125)]
    inner_left = [[-2.0, i] for i in np.linspace(-2.0, 2.0, num=125)]
    inner_right = [[2.0, i] for i in np.linspace(-2.0, 2.0, num=125)]

    outer_top = [[i, 6.0] for i in np.linspace(-6.0, 6.0, num=125)]
    outer_bot = [[i, -6.0] for i in np.linspace(-6.0, 6.0, num=125)]
    outer_left = [[-6.0, i] for i in np.linspace(-6.0, 6.0, num=125)]
    outer_right = [[6.0, i] for i in np.linspace(-6.0, 6.0, num=125)]

    X = inner_bot + inner_top + inner_left + inner_right + outer_bot + outer_top + outer_left + outer_right
    labels = [0 for i in range(500)] + [1 for i in range(500)]

    X = np.array(X)
    y = np.array(labels)

    return X, y


# Question 3c Part vi
def create_graph(dataset, r):
    neigh = NearestNeighbors(n_neighbors=r + 1)
    neigh.fit(dataset)
    NearestNeighbors(n_neighbors=r + 1)
    G_r = neigh.kneighbors_graph(dataset).toarray()
    for i in range(len(G_r)):
        G_r[i][i] = 0
    return G_r


def create_W_matrix(G_r, dataset):
    W = np.zeros(shape=(len(dataset), len(dataset)))
    for i in range(len(dataset)):
        for j in range(len(dataset)):
            if i == j:
                W[i][j] = 0
            else:
                W[i][j] = 1 if (G_r[i][j] == 1 or G_r[j][i] == 1) else 0
                W[j][i] = W[i][j]

    return W


def create_D_matrix(W, dataset):
    D = np.zeros(shape=(len(dataset), len(dataset)))
    for i in range(len(dataset)):
        for j in range(len(dataset)):
            D[i][i] += W[i][j]

    return D


def create_L_matrix(D, W):
    return D - W


def create_V_matrix(L, k):
    # w - list of eigenvalues in ascending order
    # v - 2d array of normalized eigenvectors (columns) for corresponding eigenvalues in w

    eigenvalues, eigenvectors = LA.eig(L)
    idx = eigenvalues.argsort()  # sort in ascending order
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # slice out first k columns of v - bottom k eigenvectors
    V = eigenvectors[:, 0:k]

    return V


def flexible_clustering(dataset, r, k):
    G_r = create_graph(dataset, r)
    W = create_W_matrix(G_r, dataset)
    D = create_D_matrix(W, dataset)
    L = create_L_matrix(D, W)
    V = create_V_matrix(L, k)
    cluster_centers, labels = k_means(V, k)

    return V, cluster_centers, labels


def __main__():
    data, clusters = skl_data.make_circles(n_samples=1000, noise=.01, random_state=0)
    # data, clusters = sklearn.datasets.make_moons(n_samples=1000, noise=.05)
    # data, clusters = make_rectangles()
    k = 2
    cluster_centers, labels = k_means(data, k)
    plt.scatter(data[:, 0], data[:, 1], s=5, linewidth=0.1, c=labels)
    for center in cluster_centers:
        x = center[0]
        y = center[1]
        plt.plot(x, y, color='r', marker='X')
    plt.show()

    plt.scatter(data[:, 0], data[:, 1], s=5, linewidth=0.1, c=clusters)
    plt.show()

    r_vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 800, 900]

    for r in r_vals:
        V, cluster_centers, labels = flexible_clustering(data, r, k=2)
        title = "r = " + str(r)
        plt.title(title)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.scatter(data[:, 0], data[:, 1], s=5, linewidth=0.1, c=labels)
        # plt.scatter(V[:, 0], V[:, 1], s=5, linewidth=0.1, c=labels)
        for center in cluster_centers:
            x = center[0]
            y = center[1]
            plt.plot(x, y, color='r', linewidth=0.001, marker='X')
        plt.show()

    title = "True Clustering"
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.scatter(data[:, 0], data[:, 1], s=5, linewidth=0.1, c=clusters)
    plt.show()


if __name__ == __main__():
    __main__()