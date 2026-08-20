#pragma once

#include "app/types.hpp"
#include <string>
#include <vector>

namespace ct {

struct Percentiles {
    double mean = 0;
    double p50 = 0;
    double p95 = 0;
    double p99 = 0;
};

Percentiles pct(std::vector<double> xs);
void write_bench_artifacts(const AppConfig& cfg, const std::vector<StageTimes>& measured,
                           const std::string& hw_line);

}  // namespace ct
