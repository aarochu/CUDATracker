#pragma once

#include "app/types.hpp"
#include <chrono>
#include <string>

namespace ct {

inline double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

std::string hardware_line();

}  // namespace ct
