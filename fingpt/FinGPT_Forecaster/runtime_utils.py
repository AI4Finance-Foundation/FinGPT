import os


def resolve_torch_dtype(torch_module, env=None):
    """Resolve the inference dtype for the current runtime."""
    env = os.environ if env is None else env
    dtype_name = env.get("FINGPT_TORCH_DTYPE", "auto").strip().lower()
    dtype_by_name = {
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
    }
    if hasattr(torch_module, "bfloat16"):
        dtype_by_name["bfloat16"] = torch_module.bfloat16
        dtype_by_name["bf16"] = torch_module.bfloat16

    if dtype_name in dtype_by_name:
        return dtype_by_name[dtype_name]
    if dtype_name not in {"", "auto"}:
        valid = ", ".join(sorted([*dtype_by_name, "auto"]))
        raise ValueError(f"Invalid FINGPT_TORCH_DTYPE={dtype_name!r}; expected one of: {valid}")

    return torch_module.float16 if torch_module.cuda.is_available() else torch_module.float32


def print_gpu_utilization(torch_module, nvml_module=None):
    """Print GPU memory usage when CUDA/NVML are available.

    Returns True when a GPU memory report was printed and False when reporting
    was skipped. Inference should not fail just because NVML is unavailable on a
    Windows or CPU-only runtime.
    """
    if not torch_module.cuda.is_available():
        print("CUDA is not available; skipping GPU memory report.")
        return False

    if nvml_module is None:
        try:
            import pynvml as nvml_module
        except Exception as exc:
            print(f"GPU memory report unavailable: {exc}")
            return False

    try:
        nvml_module.nvmlInit()
        handle = nvml_module.nvmlDeviceGetHandleByIndex(0)
        info = nvml_module.nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU memory occupied: {info.used // 1024**2} MB.")
        return True
    except Exception as exc:
        print(f"GPU memory report unavailable: {exc}")
        return False
    finally:
        shutdown = getattr(nvml_module, "nvmlShutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                pass


def empty_cuda_cache(torch_module):
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()
