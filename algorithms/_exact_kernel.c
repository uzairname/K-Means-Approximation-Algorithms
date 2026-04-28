/*
 * Branch-and-bound DFS kernel for exact k-means with OpenMP parallelism.
 * See algorithms/exact.py for the algorithmic justification.
 *
 * Build: cc -O3 -march=native -ffast-math -fopenmp -shared -fPIC \
 *           _exact_kernel.c -o _exact_kernel.so
 *
 * Strategy: enumerate all DFS prefixes serially down to a small depth, then
 * run an OpenMP parallel-for over those prefixes. Each thread owns its own
 * cluster state (count, sum, sumsq, cost, assignment); the only shared mutable
 * state is best_cost / best_assignment, updated rarely under a critical section.
 */

#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

typedef struct {
    int n, d, k;
    const double *X;     /* (n, d) row-major */
    const double *psq;   /* (n,) per-point squared norms */

    double best_cost;    /* shared, updated under critical */
    int   *best;         /* (n,) shared, updated under critical */
} shared_t;

typedef struct {
    long long *count;    /* (k,) */
    double    *sum;      /* (k, d) */
    double    *sumsq;    /* (k,) */
    double    *cost;     /* (k,) */
    int       *assignment; /* (n,) */
} local_t;

static local_t *local_alloc(int n, int d, int k) {
    local_t *L = (local_t *)malloc(sizeof(local_t));
    L->count      = (long long *)calloc((size_t)k, sizeof(long long));
    L->sum        = (double    *)calloc((size_t)k * d, sizeof(double));
    L->sumsq      = (double    *)calloc((size_t)k, sizeof(double));
    L->cost       = (double    *)calloc((size_t)k, sizeof(double));
    L->assignment = (int       *)calloc((size_t)n, sizeof(int));
    return L;
}

static void local_free(local_t *L) {
    free(L->count); free(L->sum); free(L->sumsq); free(L->cost); free(L->assignment);
    free(L);
}

static void local_reset(local_t *L, int n, int d, int k) {
    memset(L->count,      0, sizeof(long long) * (size_t)k);
    memset(L->sum,        0, sizeof(double)    * (size_t)k * d);
    memset(L->sumsq,      0, sizeof(double)    * (size_t)k);
    memset(L->cost,       0, sizeof(double)    * (size_t)k);
    memset(L->assignment, 0, sizeof(int)       * (size_t)n);
}

/* Apply assignment[i] = j to local state, returning (new_partial - old_partial). */
static double local_apply(local_t *L, const shared_t *S, int i, int j) {
    int d = S->d;
    const double *x = &S->X[(size_t)i * d];
    double xsq = S->psq[i];
    double old_cost = L->cost[j];

    long long c = ++L->count[j];
    double *sj = &L->sum[(size_t)j * d];
    for (int t = 0; t < d; t++) sj[t] += x[t];
    L->sumsq[j] += xsq;
    double snorm = 0.0;
    for (int t = 0; t < d; t++) snorm += sj[t] * sj[t];
    double new_cost = L->sumsq[j] - snorm / (double)c;
    L->cost[j] = new_cost;
    L->assignment[i] = j;

    return new_cost - old_cost;
}

static void local_unapply(local_t *L, const shared_t *S, int i, int j, double saved_cost) {
    int d = S->d;
    const double *x = &S->X[(size_t)i * d];
    double xsq = S->psq[i];
    L->count[j]--;
    double *sj = &L->sum[(size_t)j * d];
    for (int t = 0; t < d; t++) sj[t] -= x[t];
    L->sumsq[j] -= xsq;
    L->cost[j] = saved_cost;
}

static void dfs(shared_t *S, local_t *L, int i, double partial, int max_used) {
    /* Atomic read of shared best so stale reads are still safe (they only over-prune
     * relative to truth; pruning is monotone so correctness is preserved). */
    double bc;
    #pragma omp atomic read
    bc = S->best_cost;
    if (partial >= bc) return;

    if (i == S->n) {
        if (max_used == S->k - 1) {
            #pragma omp critical (best_update)
            {
                if (partial < S->best_cost) {
                    S->best_cost = partial;
                    memcpy(S->best, L->assignment, sizeof(int) * (size_t)S->n);
                }
            }
        }
        return;
    }

    int remaining = S->n - i;
    int clusters_left = S->k - 1 - max_used;
    if (remaining < clusters_left) return;
    int force_new = (remaining == clusters_left);

    int upper = max_used + 1;
    if (upper > S->k - 1) upper = S->k - 1;
    int lower = force_new ? upper : 0;

    for (int j = lower; j <= upper; j++) {
        double saved_cost = L->cost[j];
        double dpartial = local_apply(L, S, i, j);
        int nm = (j > max_used) ? j : max_used;

        dfs(S, L, i + 1, partial + dpartial, nm);

        local_unapply(L, S, i, j, saved_cost);
    }
}

/* ---------- prefix enumeration ---------- */

typedef struct {
    int    *assignments;   /* (capacity * depth) flat */
    int    *max_used;      /* (capacity,) */
    double *partial_cost;  /* (capacity,) */
    int     n;
    int     capacity;
    int     depth;
} prefix_list_t;

static void plist_init(prefix_list_t *PL, int depth, int initial_capacity) {
    PL->capacity     = initial_capacity;
    PL->depth        = depth;
    PL->n            = 0;
    PL->assignments  = (int    *)malloc(sizeof(int)    * (size_t)PL->capacity * depth);
    PL->max_used     = (int    *)malloc(sizeof(int)    * (size_t)PL->capacity);
    PL->partial_cost = (double *)malloc(sizeof(double) * (size_t)PL->capacity);
}

static void plist_grow(prefix_list_t *PL) {
    PL->capacity     *= 2;
    PL->assignments   = (int    *)realloc(PL->assignments,  sizeof(int)    * (size_t)PL->capacity * PL->depth);
    PL->max_used      = (int    *)realloc(PL->max_used,     sizeof(int)    * (size_t)PL->capacity);
    PL->partial_cost  = (double *)realloc(PL->partial_cost, sizeof(double) * (size_t)PL->capacity);
}

static void plist_free(prefix_list_t *PL) {
    free(PL->assignments); free(PL->max_used); free(PL->partial_cost);
}

static void enum_prefixes(shared_t *S, local_t *L, int i, double partial,
                          int max_used, prefix_list_t *PL) {
    if (i == PL->depth) {
        if (PL->n >= PL->capacity) plist_grow(PL);
        memcpy(&PL->assignments[(size_t)PL->n * PL->depth],
               L->assignment, sizeof(int) * (size_t)PL->depth);
        PL->max_used[PL->n]     = max_used;
        PL->partial_cost[PL->n] = partial;
        PL->n++;
        return;
    }

    /* The forced-new feasibility check (remaining < clusters_left) and force_new
     * apply at every depth, including during prefix enumeration. */
    int remaining = S->n - i;
    int clusters_left = S->k - 1 - max_used;
    if (remaining < clusters_left) return;
    int force_new = (remaining == clusters_left);

    int upper = max_used + 1;
    if (upper > S->k - 1) upper = S->k - 1;
    int lower = force_new ? upper : 0;

    for (int j = lower; j <= upper; j++) {
        double saved_cost = L->cost[j];
        double dpartial = local_apply(L, S, i, j);
        int nm = (j > max_used) ? j : max_used;

        enum_prefixes(S, L, i + 1, partial + dpartial, nm, PL);

        local_unapply(L, S, i, j, saved_cost);
    }
}

/* ---------- entry point ---------- */

void exact_kmeans_c(
    const double *X,
    int n, int d, int k,
    double initial_best,
    int *assignment,
    double *out_cost
) {
    shared_t S;
    S.n = n; S.d = d; S.k = k; S.X = X;
    S.best_cost = initial_best;

    double *psq = (double *)malloc(sizeof(double) * (size_t)n);
    for (int i = 0; i < n; i++) {
        double s = 0.0;
        for (int t = 0; t < d; t++) {
            double v = X[(size_t)i * d + t];
            s += v * v;
        }
        psq[i] = s;
    }
    S.psq = psq;
    S.best = (int *)malloc(sizeof(int) * (size_t)n);
    memcpy(S.best, assignment, sizeof(int) * (size_t)n);

    /* Pick prefix depth. We want enough prefixes to keep all threads busy with
     * dynamic scheduling, but not so deep that prefix enumeration itself dominates.
     * Bell numbers: depth 6 -> 203 partitions (capped by k); depth 7 -> 877. */
    int n_threads = 1;
    #ifdef _OPENMP
    n_threads = omp_get_max_threads();
    #endif
    int depth = 6;
    if (depth > n) depth = n;
    if (n_threads >= 32 && depth < 7 && n >= 7) depth = 7;

    /* Phase 1: serial prefix enumeration */
    local_t *L0 = local_alloc(n, d, k);
    prefix_list_t PL;
    plist_init(&PL, depth, 1024);
    enum_prefixes(&S, L0, 0, 0.0, -1, &PL);
    local_free(L0);

    /* Phase 2: parallel DFS over prefixes */
    if (PL.n > 0) {
        #pragma omp parallel
        {
            local_t *L = local_alloc(n, d, k);
            #pragma omp for schedule(dynamic, 1) nowait
            for (int p = 0; p < PL.n; p++) {
                /* quick prune against current best before initializing */
                double bc;
                #pragma omp atomic read
                bc = S.best_cost;
                if (PL.partial_cost[p] >= bc) continue;

                local_reset(L, n, d, k);
                int *prefix = &PL.assignments[(size_t)p * depth];
                for (int i = 0; i < depth; i++) {
                    local_apply(L, &S, i, prefix[i]);
                }
                dfs(&S, L, depth, PL.partial_cost[p], PL.max_used[p]);
            }
            local_free(L);
        }
    }

    memcpy(assignment, S.best, sizeof(int) * (size_t)n);
    *out_cost = S.best_cost;

    free(psq);
    free(S.best);
    plist_free(&PL);
}
