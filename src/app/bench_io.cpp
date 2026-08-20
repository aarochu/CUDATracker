#include "bench_io.hpp"
#include "telemetry/profiler.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <vector>

namespace ct {
namespace {

double qtile(const std::vector<double>& s, double p) {
    if (s.empty()) return 0;
    const double k = (s.size() - 1) * (p / 100.0);
    const size_t f = static_cast<size_t>(std::floor(k));
    const size_t c = static_cast<size_t>(std::ceil(k));
    if (f == c) return s[f];
    return s[f] * (c - k) + s[c] * (k - f);
}

std::string json_escape(const std::string& s) {
    std::string o;
    o.reserve(s.size());
    for (char ch : s) {
        if (ch == '\\' || ch == '"') {
            o.push_back('\\');
            o.push_back(ch);
        } else {
            o.push_back(ch);
        }
    }
    return o;
}

}  // namespace

Percentiles pct(std::vector<double> xs) {
    Percentiles p;
    if (xs.empty()) return p;
    double sum = 0;
    for (double v : xs) sum += v;
    p.mean = sum / static_cast<double>(xs.size());
    std::sort(xs.begin(), xs.end());
    p.p50 = qtile(xs, 50);
    p.p95 = qtile(xs, 95);
    p.p99 = qtile(xs, 99);
    return p;
}

void write_bench_artifacts(const AppConfig& cfg, const std::vector<StageTimes>& measured,
                           const std::string& hw_line) {
    std::vector<double> e2e, pre, xfer, inf, post, trk, cap, rend;
    e2e.reserve(measured.size());
    for (const auto& t : measured) {
        cap.push_back(t.capture_ms);
        pre.push_back(t.preprocess_ms);
        xfer.push_back(t.transfer_ms);
        inf.push_back(t.inference_ms);
        post.push_back(t.postprocess_ms);
        trk.push_back(t.tracking_ms);
        rend.push_back(t.render_ms);
        e2e.push_back(t.end_to_end_ms);
    }
    const auto e = pct(e2e);
    const auto ppre = pct(pre);
    const auto px = pct(xfer);
    const auto pi = pct(inf);
    const auto pp = pct(post);
    const auto pt = pct(trk);
    double wall = 0;
    for (double v : e2e) wall += v;
    const double fps = wall > 0 ? (measured.size() / (wall / 1000.0)) : 0;

    std::string path = cfg.output_metrics;
    if (path.empty()) path = "benchmarks/results/last_run.json";
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());

    std::ofstream js(path);
    if (!js) {
        log_line("ERROR", "bench", "cannot write " + path);
        return;
    }
    js << "{\n";
    js << "  \"config\": {\n";
    js << "    \"source\": \"" << json_escape(cfg.source) << "\",\n";
    js << "    \"preprocess\": \"" << json_escape(cfg.preprocess) << "\",\n";
    js << "    \"backend\": \"" << json_escape(cfg.backend) << "\",\n";
    js << "    \"precision\": \"" << json_escape(cfg.precision) << "\",\n";
    js << "    \"imgsz\": " << cfg.imgsz << "\n";
    js << "  },\n";
    js << "  \"hardware\": \"" << json_escape(hw_line) << "\",\n";
    js << "  \"summary\": {\n";
    js << "    \"fps\": " << fps << ",\n";
    js << "    \"frames\": " << measured.size() << ",\n";
    js << "    \"end_to_end\": {\"mean\": " << e.mean << ", \"p50\": " << e.p50 << ", \"p95\": " << e.p95
       << ", \"p99\": " << e.p99 << "},\n";
    js << "    \"stages\": {\n";
    js << "      \"capture\": {\"mean\": " << pct(cap).mean << "},\n";
    js << "      \"preprocess\": {\"mean\": " << ppre.mean << "},\n";
    js << "      \"transfer\": {\"mean\": " << px.mean << "},\n";
    js << "      \"inference\": {\"mean\": " << pi.mean << "},\n";
    js << "      \"postprocess\": {\"mean\": " << pp.mean << "},\n";
    js << "      \"tracking\": {\"mean\": " << pt.mean << "},\n";
    js << "      \"render\": {\"mean\": " << pct(rend).mean << "}\n";
    js << "    }\n";
    js << "  }\n";
    js << "}\n";
    js.close();

    const std::string csv_path = path.substr(0, path.find_last_of('.')) + ".csv";
    const bool fresh = std::ifstream(csv_path).fail();
    std::ofstream csv(csv_path, std::ios::app);
    if (fresh) {
        csv << "preprocess,backend,precision,resolution,model,fps,e2e_mean_ms,e2e_p50_ms,e2e_p95_ms,e2e_p99_ms,"
               "preprocess_ms,transfer_ms,inference_ms,postprocess_ms,tracking_ms,skipped,skip_reason\n";
    }
    csv << cfg.preprocess << ',' << cfg.backend << ',' << cfg.precision << ',' << cfg.width << 'x' << cfg.height
        << ',' << cfg.model_name << ',' << fps << ',' << e.mean << ',' << e.p50 << ',' << e.p95 << ',' << e.p99 << ','
        << ppre.mean << ',' << px.mean << ',' << pi.mean << ',' << pp.mean << ',' << pt.mean << ",false,\n";
    log_line("INFO", "bench", "wrote " + path + " fps=" + std::to_string(fps));
}

}  // namespace ct
