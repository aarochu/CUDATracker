#include "types.hpp"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace ct {

void log_line(const char* level, const char* stage, const std::string& msg) {
    using clock = std::chrono::system_clock;
    auto now = clock::now();
    auto t = clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    std::ostream& os = (std::string(level) == "ERROR") ? std::cerr : std::cout;
    os << std::put_time(&tm, "%H:%M:%S") << '.' << std::setfill('0') << std::setw(3) << ms.count()
       << ' ' << level << " [" << stage << "] " << msg << std::endl;
}

}  // namespace ct
