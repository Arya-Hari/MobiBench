# src/system_monitor.py

import psutil, time

class SystemMonitor:
    def __init__(self, pid):
        self.pid = pid
        self.process = psutil.Process(pid)
        self.history = []

    def sample(self):
        try:
            cpu = self.process.cpu_percent(interval=0.1)
            mem = self.process.memory_info().rss / 1e6  # MB
            energy = None  # placeholder for RAPL / NVIDIA SMI integration
            self.history.append({
                "time": time.time(),
                "cpu_percent": cpu,
                "memory_mb": mem,
                "energy_j": energy
            })
        except psutil.NoSuchProcess:
            return False
        return True

    def run(self, interval=0.5):
        while True:
            if not self.sample():
                break
            time.sleep(interval)

    def summary(self):
        if not self.history:
            return {}
        avg_cpu = sum(h["cpu_percent"] for h in self.history) / len(self.history)
        peak_mem = max(h["memory_mb"] for h in self.history)
        return {
            "avg_cpu_percent": avg_cpu,
            "peak_memory_mb": peak_mem,
            "samples": len(self.history)
        }
