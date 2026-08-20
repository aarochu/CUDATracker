from cudatracker.telemetry.hardware import parse_tegrastats_line


def test_tegrastats_gr3d_and_vin():
    line = (
        "RAM 1023/7777MB CPU [1%@2201] EMC_FREQ 0% GR3D_FREQ 76%@[1147] "
        "GPU@41C VDD_GPU_SOC 2178mW/2178mW VDD_CPU_CV 845mW/845mW VIN_SYS_5V0 4621mW/4621mW"
    )
    p = parse_tegrastats_line(line)
    assert p["gpu_util"] == 76.0
    assert abs(p["power_w"] - 4.621) < 1e-6


def test_tegrastats_prefers_vdd_in_over_sum():
    line = "GR3D_FREQ 10% VDD_GPU_SOC 2000mW VDD_IN 5000mW"
    p = parse_tegrastats_line(line)
    assert p["gpu_util"] == 10.0
    assert abs(p["power_w"] - 5.0) < 1e-6


def test_tegrastats_pom_5v_in():
    line = "GR3D_FREQ 3% POM_5V_IN 1234mW"
    p = parse_tegrastats_line(line)
    assert abs(p["power_w"] - 1.234) < 1e-6
    assert p["gpu_util"] == 3.0
