from cudatracker.types import StageTimes


def test_slowest_is_inference():
    t = StageTimes(
        capture_ms=1.8,
        preprocess_ms=3.7,
        transfer_ms=0.4,
        inference_ms=14.2,
        postprocess_ms=0.8,
        tracking_ms=1.1,
        render_ms=2.0,
        end_to_end_ms=24.0,
        inference_gpu_ms=13.8,
        preprocess_gpu_ms=3.1,
    )
    t.finish_totals()
    name, ms = t.slowest()
    assert name == "inference"
    assert abs(ms - 14.2) < 1e-6
    assert "TOTAL" in "\n".join(t.table_lines())
    assert "bottleneck inference" in t.table_lines()[-1]


def test_cpu_wait_split():
    t = StageTimes(end_to_end_ms=20.0, inference_sync_ms=12.0, preprocess_sync_ms=1.0)
    t.finish_totals()
    assert abs(t.sync_ms - 13.0) < 1e-6
    assert abs(t.cpu_ms - 7.0) < 1e-6
