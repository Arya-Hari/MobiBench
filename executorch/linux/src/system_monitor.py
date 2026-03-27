import psutil
import time

try:
    import pynvml
    NVML_AVAILABLE = True
    pynvml.nvmlInit()
except ImportError:
    NVML_AVAILABLE = False


class SystemMonitor:
    def __init__(self, pid):
        self.pid = pid
        self.process = psutil.Process(pid)
        self.history = []
        self.gpu_available = NVML_AVAILABLE
        self._running = True  # flag to control monitoring loop

    def sample(self):
        """Take one system metrics sample."""
        try:
            cpu = self.process.cpu_percent(interval=0.1)
            mem = self.process.memory_info().rss / 1e6  # MB
            energy = None  # placeholder for RAPL integration
            gpu_util = None
            gpu_mem = None

            if self.gpu_available:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                    gpu_mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_mem = gpu_mem_info.used / 1e6  # MB
                except Exception:
                    pass

            self.history.append({
                "time": time.time(),
                "cpu_percent": cpu,
                "memory_mb": mem,
                "gpu_util_percent": gpu_util,
                "gpu_memory_mb": gpu_mem,
                "energy_j": energy
            })
        except psutil.NoSuchProcess:
            return False
        return True

    def run(self, interval=0.5):
        """Background monitoring loop until stopped."""
        while self._running:
            if not self.sample():
                break
            time.sleep(interval)

    def stop(self):
        """Stop the monitoring loop."""
        self._running = False

    def summary(self):
        """Return aggregate stats of all collected samples."""
        if not self.history:
            return {}

        avg_cpu = sum(h["cpu_percent"] for h in self.history) / len(self.history)
        peak_mem = max(h["memory_mb"] for h in self.history)

        summary = {
            "avg_cpu_percent": avg_cpu,
            "peak_memory_mb": peak_mem,
            "samples": len(self.history)
        }

        # Add GPU metrics if available
        gpu_utils = [h["gpu_util_percent"] for h in self.history if h["gpu_util_percent"] is not None]
        if gpu_utils:
            summary["avg_gpu_util_percent"] = sum(gpu_utils) / len(gpu_utils)

        gpu_mems = [h["gpu_memory_mb"] for h in self.history if h["gpu_memory_mb"] is not None]
        if gpu_mems:
            summary["peak_gpu_memory_mb"] = max(gpu_mems)

        return summary
