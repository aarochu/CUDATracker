from __future__ import annotations

import numpy as np


def linear_sum_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hungarian algorithm. Returns row_ind, col_ind minimizing sum(cost[r,c])."""
    cost = np.asarray(cost, dtype=np.float64)
    if cost.size == 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32)
    n, m = cost.shape
    dim = max(n, m)
    # Dummy rows/cols must be more expensive than any real 1-IoU cost in [0, 1].
    # Padding with 0 ties a perfect match and can assign the wrong track.
    pad = np.full((dim, dim), 1.0e5, dtype=np.float64)
    pad[:n, :m] = cost
    u = np.zeros(dim + 1)
    v = np.zeros(dim + 1)
    p = np.zeros(dim + 1, dtype=np.int32)
    way = np.zeros(dim + 1, dtype=np.int32)
    for i in range(1, dim + 1):
        p[0] = i
        j0 = 0
        minv = np.full(dim + 1, np.inf)
        used = np.zeros(dim + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, dim + 1):
                if used[j]:
                    continue
                cur = pad[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(dim + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    col_of_row = np.full(n, -1, dtype=np.int32)
    for j in range(1, dim + 1):
        i = p[j]
        if 1 <= i <= n and 1 <= j <= m:
            col_of_row[i - 1] = j - 1
    rows = np.arange(n, dtype=np.int32)
    cols = col_of_row
    valid = cols >= 0
    return rows[valid], cols[valid]
