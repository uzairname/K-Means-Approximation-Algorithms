import numpy as np


def k_means_plus_plus(X, k):
    n_samples, n_features = X.shape
    centroids = np.empty((k, n_features), dtype=X.dtype)

    centroids[0] = X[np.random.choice(n_samples)]

    diff = X - centroids[0]
    closest_sq = np.einsum("ij,ij->i", diff, diff)

    for i in range(1, k):
        total = closest_sq.sum()
        if total <= 0:
            centroids[i] = X[np.random.choice(n_samples)]
        else:
            centroids[i] = X[np.random.choice(n_samples, p=closest_sq / total)]
        diff = X - centroids[i]
        new_sq = np.einsum("ij,ij->i", diff, diff)
        np.minimum(closest_sq, new_sq, out=closest_sq)

    return centroids

def lloyds_algorithm(X, k, max_iters=100, tol=1e-5):
    """
    Lloyds algo with k means++

    Returns: centers, labels, wcss
    """
    n_samples, n_features = X.shape
    centroids = k_means_plus_plus(X, k)
    x_norm = np.einsum("ij,ij->i", X, X)
    wcss = np.empty(max_iters)
    labels = np.empty(n_samples, dtype=np.intp)

    for i in range(max_iters):
        c_norm = np.einsum("ij,ij->i", centroids, centroids)
        d2 = x_norm[:, None] - 2.0 * (X @ centroids.T) + c_norm[None, :]
        labels = np.argmin(d2, axis=1)
        wcss[i] = max(0.0, float(d2[np.arange(n_samples), labels].sum()))

        new_centroids = np.zeros_like(centroids)
        np.add.at(new_centroids, labels, X)
        counts = np.bincount(labels, minlength=k)
        nz = counts > 0
        new_centroids[nz] /= counts[nz, None]

        new_centroids[~nz] = centroids[~nz]

        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if shift < tol:
            break

    return centroids, labels, wcss[:i+1]
