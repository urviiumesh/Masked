from api.services.runtime_device import (
    PROVIDER_PRIORITY,
    ctx_id_for_provider,
    is_gpu_active,
    runtime_info,
    select_providers,
)


def test_provider_priority_ends_with_cpu():
    assert PROVIDER_PRIORITY[-1] == "CPUExecutionProvider"
    assert "CUDAExecutionProvider" in PROVIDER_PRIORITY
    assert "DmlExecutionProvider" in PROVIDER_PRIORITY


def test_select_providers_includes_cpu():
    providers = select_providers()
    assert "CPUExecutionProvider" in providers
    assert len(providers) >= 1


def test_ctx_id_cpu_is_negative():
    assert ctx_id_for_provider("CPUExecutionProvider") == -1
    assert ctx_id_for_provider("CUDAExecutionProvider") == 0


def test_runtime_info_shape():
    info = runtime_info()
    assert "active_provider" in info
    assert "available_providers" in info
    assert "gpu_enabled" in info
    assert info["gpu_enabled"] == (info["active_provider"] != "CPUExecutionProvider")
    assert info["gpu_enabled"] == is_gpu_active()


def test_stream_det_sizes():
    from register_face import stream_det_size, video_det_size

    assert stream_det_size() == (960, 960)
    assert video_det_size() == (640, 640)
