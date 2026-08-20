#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace ct {

// Hungarian (min-cost). Dummy cells must already be filled with a large cost.
inline std::pair<std::vector<int>, std::vector<int>> linear_sum_assignment(
    const std::vector<std::vector<double>>& cost) {
    const int n = static_cast<int>(cost.size());
    const int m = n ? static_cast<int>(cost[0].size()) : 0;
    if (n == 0 || m == 0) return {{}, {}};
    const int dim = std::max(n, m);
    const double kDummy = 1.0e5;
    std::vector<std::vector<double>> pad(dim, std::vector<double>(dim, kDummy));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j) pad[i][j] = cost[i][j];

    std::vector<double> u(dim + 1, 0.0), v(dim + 1, 0.0);
    std::vector<int> p(dim + 1, 0), way(dim + 1, 0);
    for (int i = 1; i <= dim; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(dim + 1, std::numeric_limits<double>::infinity());
        std::vector<char> used(dim + 1, 0);
        while (true) {
            used[j0] = 1;
            const int i0 = p[j0];
            double delta = std::numeric_limits<double>::infinity();
            int j1 = 0;
            for (int j = 1; j <= dim; ++j) {
                if (used[j]) continue;
                const double cur = pad[i0 - 1][j - 1] - u[i0] - v[j];
                if (cur < minv[j]) {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if (minv[j] < delta) {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for (int j = 0; j <= dim; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if (p[j0] == 0) break;
        }
        while (true) {
            const int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if (j0 == 0) break;
        }
    }
    std::vector<int> rows, cols;
    for (int j = 1; j <= dim; ++j) {
        const int i = p[j];
        if (i >= 1 && i <= n && j >= 1 && j <= m) {
            rows.push_back(i - 1);
            cols.push_back(j - 1);
        }
    }
    return {rows, cols};
}

}  // namespace ct
