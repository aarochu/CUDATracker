#include "profiler.hpp"

#include <fstream>
#include <sstream>

#ifdef _WIN32
#include <windows.h>
#endif

namespace ct {

std::string hardware_line() {
    std::ostringstream os;
#ifdef __linux__
    std::ifstream stat("/proc/stat");
    std::string cpu;
    long user = 0, nice = 0, sys = 0, idle = 0;
    if (stat) stat >> cpu >> user >> nice >> sys >> idle;
    os << "cpu ticks " << (user + sys);
    std::ifstream mem("/proc/meminfo");
    std::string k;
    long mem_total = 0, mem_avail = 0;
    while (mem >> k) {
        long v;
        std::string unit;
        mem >> v >> unit;
        if (k == "MemTotal:") mem_total = v;
        if (k == "MemAvailable:") mem_avail = v;
    }
    if (mem_total) os << "  ram_used_kB " << (mem_total - mem_avail);
#else
    MEMORYSTATUSEX s{};
    s.dwLength = sizeof(s);
    if (GlobalMemoryStatusEx(&s)) {
        os << "ram_used_MB " << int((s.ullTotalPhys - s.ullAvailPhys) / (1024 * 1024));
    }
#endif
    return os.str();
}

}  // namespace ct
