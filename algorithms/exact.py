import ctypes
import multiprocessing as mp
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from algorithms.lloyd import lloyds_algorithm


_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "_exact_kernel.c")
_LIB = os.path.join(_HERE, "_exact_kernel.so")


def _build_kernel():
    needs_build = (
        not os.path.exists(_LIB)
        or os.path.getmtime(_LIB) < os.path.getmtime(_SRC)
    )
    if needs_build:
        cc = os.environ.get("CC", "cc")
        cmd = [
            cc, "-O3", "-march=native", "-ffast-math", "-fopenmp",
            "-shared", "-fPIC",
            _SRC, "-o", _LIB,
        ]
        subprocess.run(cmd, check=True)

    lib = ctypes.CDLL(_LIB)
    lib.exact_kmeans_c.restype = None
    lib.exact_kmeans_c.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # X
        ctypes.c_int,                      # n
        ctypes.c_int,                      # d
        ctypes.c_int,                      # k
        ctypes.c_double,                   # initial_best
        ctypes.POINTER(ctypes.c_int),      # assignment (in/out)
        ctypes.POINTER(ctypes.c_double),   # out_cost
    ]
    return lib


_LIB_HANDLE = _build_kernel()


def _lloyd_warm_start(X, k, n_restarts=10, max_iters=100):
    best_cost = np.inf
    best_labels = None
    best_centroids = None
    for _ in range(n_restarts):
        centroids, labels, wcss = lloyds_algorithm(X, k, max_iters=max_iters)
        cost = wcss[-1]
        if cost < best_cost:
            best_cost = cost
            best_labels = labels
            best_centroids = centroids
    return best_centroids, best_labels, float(best_cost)


def _finalize(X, labels, fallback_centroids, k):
    _, d = X.shape
    centroids = np.empty((k, d))
    for j in range(k):
        mask = labels == j
        if np.any(mask):
            centroids[j] = X[mask].mean(axis=0)
        else:
            centroids[j] = fallback_centroids[j]
    return centroids


def exact_kmeans(X, k, warm_restarts=10):
    X = np.ascontiguousarray(X, dtype=np.float64)
    n, d = X.shape
    if k <= 0 or k > n:
        raise ValueError(f"need 1 <= k <= n, got k={k}, n={n}")

    centroids_ws, labels_ws, best_cost = _lloyd_warm_start(X, k, n_restarts=warm_restarts)
    assignment = np.ascontiguousarray(labels_ws, dtype=np.int32)
    out_cost = ctypes.c_double(0.0)

    _LIB_HANDLE.exact_kmeans_c(
        X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(n),
        ctypes.c_int(d),
        ctypes.c_int(k),
        ctypes.c_double(best_cost),
        assignment.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.byref(out_cost),
    )

    labels = assignment.astype(np.int64)
    centroids = _finalize(X, labels, centroids_ws, k)
    return centroids, labels, float(out_cost.value)


def _bench_worker(kwargs):
    from algorithms.exact import exact_kmeans
    return exact_kmeans(**kwargs)


def bench_exact_kmeans(jobs, n_workers=None, omp_threads=1):
    """
    Run many exact_kmeans calls in parallel via a process pool.
    """
    if n_workers is None:
        n_workers = os.cpu_count() or 1

    job_args = []
    for j in jobs:
        if isinstance(j, dict):
            job_args.append(j)
        else:
            X, k = j
            job_args.append({"X": X, "k": k})

    ctx = mp.get_context("spawn")
    prev = os.environ.get("OMP_NUM_THREADS")
    os.environ["OMP_NUM_THREADS"] = str(omp_threads)
    try:
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
            return list(ex.map(_bench_worker, job_args))
    finally:
        if prev is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = prev
