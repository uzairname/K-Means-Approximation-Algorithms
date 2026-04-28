import numpy as np

from algorithms.lloyd import k_means_plus_plus


def _wcss(X, centers):
    d2 = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return d2.min(axis=1).sum()


def local_search_kmeans(X, k, max_swaps=None):
    n = X.shape[0]
    centers = k_means_plus_plus(X, k).copy()
    history = [_wcss(X, centers)]
    cap = max_swaps if max_swaps is not None else 50 * k

    for _ in range(cap):
        cur = history[-1]
        best_delta = 0.0
        best_ci, best_p = -1, -1

        for ci in range(k):
            for p in range(n):
                if np.any(np.all(centers == X[p], axis=1)):
                    continue
                trial = centers.copy()
                trial[ci] = X[p]
                delta = _wcss(X, trial) - cur
                if delta < -1e-12 and delta < best_delta:
                    best_delta, best_ci, best_p = delta, ci, p

        if best_ci == -1:
            break
        centers = centers.copy()
        centers[best_ci] = X[best_p]
        history.append(cur + best_delta)

    d2 = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    labels = np.argmin(d2, axis=1)
    return centers, labels, history
