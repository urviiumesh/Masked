import os
import platform
from functools import lru_cache
from typing import Any


PROVIDER_PRIORITY = [
    "CUDAExecutionProvider",
    "TensorrtExecutionProvider",
    "DmlExecutionProvider",
    "OpenVINOExecutionProvider",
    "CoreMLExecutionProvider",
    "CPUExecutionProvider",
]


def _env_force() -> str | None:
    forced = os.environ.get("DHRISHTI_ORT_PROVIDER", "").strip()
    return forced or None


def _require_gpu() -> bool:
    return os.environ.get("DHRISHTI_REQUIRE_GPU", "0").strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def available_providers() -> list[str]:
    try:
        import onnxruntime as ort
        return list(ort.get_available_providers())
    except Exception:
        return ["CPUExecutionProvider"]


@lru_cache(maxsize=1)
def select_providers() -> list[str]:
    forced = _env_force()
    available = available_providers()
    if forced:
        if forced in available:
            return [forced, "CPUExecutionProvider"] if forced != "CPUExecutionProvider" else ["CPUExecutionProvider"]
        if _require_gpu():
            raise RuntimeError(
                f"Required ONNX Runtime GPU provider {forced!r} is unavailable. "
                f"Available providers: {available}"
            )
        return ["CPUExecutionProvider"]

    chosen: list[str] = []
    for provider in PROVIDER_PRIORITY:
        if provider in available and provider not in chosen:
            chosen.append(provider)
    if "CPUExecutionProvider" not in chosen:
        chosen.append("CPUExecutionProvider")
    if not chosen:
        if _require_gpu():
            raise RuntimeError(f"GPU execution is required, but no GPU provider is available: {available}")
        chosen = ["CPUExecutionProvider"]
    return chosen


@lru_cache(maxsize=1)
def active_provider() -> str:
    return select_providers()[0]


def is_gpu_active() -> bool:
    return active_provider() != "CPUExecutionProvider"


def ctx_id_for_provider(provider: str | None = None) -> int:
    provider = provider or active_provider()
    if provider == "CPUExecutionProvider":
        return -1
    return 0


def provider_options() -> list[dict[str, Any]] | None:
    providers = select_providers()
    options: list[dict[str, Any]] = []
    for provider in providers:
        if provider == "CUDAExecutionProvider":
            options.append({
                "device_id": "0",
                "arena_extend_strategy": "kNextPowerOfTwo",
                "gpu_mem_limit": str(2 * 1024 * 1024 * 1024),
                "cudnn_conv_algo_search": "HEURISTIC",
                "do_copy_in_default_stream": "1",
            })
        elif provider == "DmlExecutionProvider":
            options.append({"device_id": 0})
        elif provider == "OpenVINOExecutionProvider":
            options.append({"device_type": "GPU"})
        else:
            options.append({})
    return options


def runtime_info() -> dict[str, Any]:
    providers = select_providers()
    active = providers[0]
    return {
        "platform": platform.system(),
        "available_providers": available_providers(),
        "selected_providers": providers,
        "active_provider": active,
        "gpu_enabled": active != "CPUExecutionProvider",
        "ctx_id": ctx_id_for_provider(active),
        "hint": _install_hint(active, available_providers()),
    }


def _install_hint(active: str, available: list[str]) -> str:
    if active != "CPUExecutionProvider":
        return f"Using {active}"
    system = platform.system().lower()
    if system == "windows":
        return "Install onnxruntime-directml for Intel/AMD/NVIDIA GPU on Windows, or onnxruntime-gpu for NVIDIA CUDA"
    return "Install onnxruntime-gpu for NVIDIA CUDA, or onnxruntime-openvino for Intel GPU on Linux"


def face_analysis_kwargs() -> dict[str, Any]:
    providers = select_providers()
    kwargs: dict[str, Any] = {"providers": providers}
    if "CUDAExecutionProvider" in providers:
        kwargs["provider_options"] = provider_options()
    return kwargs
