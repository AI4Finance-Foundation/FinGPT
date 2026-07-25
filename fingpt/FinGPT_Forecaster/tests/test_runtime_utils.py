import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime_utils import empty_cuda_cache, print_gpu_utilization, resolve_torch_dtype


class FakeCuda:
    def __init__(self, available):
        self.available = available
        self.empty_cache_called = False

    def is_available(self):
        return self.available

    def empty_cache(self):
        self.empty_cache_called = True


class FakeTorch:
    float16 = "float16"
    float32 = "float32"
    bfloat16 = "bfloat16"

    def __init__(self, cuda_available):
        self.cuda = FakeCuda(cuda_available)


class FakeMemoryInfo:
    used = 512 * 1024**2


class FakeNvml:
    def __init__(self):
        self.shutdown_called = False

    def nvmlInit(self):
        return None

    def nvmlDeviceGetHandleByIndex(self, index):
        return f"gpu-{index}"

    def nvmlDeviceGetMemoryInfo(self, handle):
        return FakeMemoryInfo()

    def nvmlShutdown(self):
        self.shutdown_called = True


class RuntimeUtilsTest(unittest.TestCase):
    def test_resolve_dtype_uses_float16_on_cuda(self):
        self.assertEqual(resolve_torch_dtype(FakeTorch(cuda_available=True), env={}), "float16")

    def test_resolve_dtype_uses_float32_without_cuda(self):
        self.assertEqual(resolve_torch_dtype(FakeTorch(cuda_available=False), env={}), "float32")

    def test_resolve_dtype_respects_override(self):
        torch_module = FakeTorch(cuda_available=True)
        self.assertEqual(resolve_torch_dtype(torch_module, env={"FINGPT_TORCH_DTYPE": "fp32"}), "float32")
        self.assertEqual(resolve_torch_dtype(torch_module, env={"FINGPT_TORCH_DTYPE": "bf16"}), "bfloat16")

    def test_resolve_dtype_rejects_unknown_override(self):
        with self.assertRaisesRegex(ValueError, "FINGPT_TORCH_DTYPE"):
            resolve_torch_dtype(FakeTorch(cuda_available=True), env={"FINGPT_TORCH_DTYPE": "fp15"})

    def test_print_gpu_utilization_skips_cpu_only_runtime(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            printed = print_gpu_utilization(FakeTorch(cuda_available=False), nvml_module=FakeNvml())

        self.assertFalse(printed)
        self.assertIn("CUDA is not available", stdout.getvalue())

    def test_print_gpu_utilization_handles_nvml_failure(self):
        class BrokenNvml(FakeNvml):
            def nvmlInit(self):
                raise RuntimeError("NVML Shared Library Not Found")

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            printed = print_gpu_utilization(FakeTorch(cuda_available=True), nvml_module=BrokenNvml())

        self.assertFalse(printed)
        self.assertIn("GPU memory report unavailable", stdout.getvalue())

    def test_print_gpu_utilization_reports_and_shuts_down_nvml(self):
        nvml = FakeNvml()
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            printed = print_gpu_utilization(FakeTorch(cuda_available=True), nvml_module=nvml)

        self.assertTrue(printed)
        self.assertTrue(nvml.shutdown_called)
        self.assertIn("GPU memory occupied: 512 MB.", stdout.getvalue())

    def test_empty_cuda_cache_only_when_cuda_is_available(self):
        gpu_torch = FakeTorch(cuda_available=True)
        cpu_torch = FakeTorch(cuda_available=False)

        empty_cuda_cache(gpu_torch)
        empty_cuda_cache(cpu_torch)

        self.assertTrue(gpu_torch.cuda.empty_cache_called)
        self.assertFalse(cpu_torch.cuda.empty_cache_called)


if __name__ == "__main__":
    unittest.main()
