import numpy as np

from algorithms.lloyd import k_means_plus_plus


def lightweight_coreset(X, m, rng=None):
    """
    Lightweight coreset (Bachem, Lucic, Krause 2018).
    """
    rng = np.random.default_rng() if rng is None else rng
    n = X.shape[0]
    mu = X.mean(axis=0)
    d2 = np.sum((X - mu) ** 2, axis=1)
    q = 0.5 / n + 0.5 * d2 / d2.sum()
    idx = rng.choice(n, size=m, replace=True, p=q)
    w = 1.0 / (m * q[idx])
    return X[idx], w


def weighted_lloyd(S, w, k, max_iters=100, tol=1e-5):
    """Lloyd's algorithm on a weighted point set, k-means++ init."""
    centroids = k_means_plus_plus(S, k)
    for _ in range(max_iters):
        d2 = np.sum((S[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = np.argmin(d2, axis=1)
        new = np.empty_like(centroids)
        for j in range(k):
            mask = labels == j
            if not np.any(mask):
                new[j] = centroids[j]
                continue
            ws = w[mask]
            new[j] = (S[mask] * ws[:, None]).sum(axis=0) / ws.sum()
        if np.linalg.norm(new - centroids) < tol:
            centroids = new
            break
        centroids = new
    return centroids


def coreset_kmeans(X, k, m, rng=None):
    S, w = lightweight_coreset(X, m, rng=rng)
    centers = weighted_lloyd(S, w, k)
    d2 = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    labels = np.argmin(d2, axis=1)
    cost = d2[np.arange(X.shape[0]), labels].sum()
    return centers, labels, [cost]
