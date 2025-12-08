import json
import statistics

# === CONFIG ===
# Path to your JSON results file
path = "/Users/rohitsuresh/Downloads/honeypot/summarization_tinyllama_inteli5.json"

# === LOAD DATA ===
with open(path, "r") as f:
    data = json.load(f)

# === INITIALIZE ===
wall_times = []
prefill_tps = []
decode_tps = []
cpu_usage = []
memory_usage = []

# === EXTRACT METRICS ===
for entry in data:
    # Timing metrics
    if "wallclock_s" in entry:
        wall_times.append(entry["wallclock_s"])

    if "timings" in entry:
        timings = entry["timings"]
        if timings.get("prefill_tps") is not None:
            prefill_tps.append(timings["prefill_tps"])
        if timings.get("decode_tps") is not None:
            decode_tps.append(timings["decode_tps"])

    # System metrics
    if "system_metrics" in entry:
        sys = entry["system_metrics"]
        if sys.get("avg_cpu_percent") is not None:
            cpu_usage.append(sys["avg_cpu_percent"])
        if sys.get("peak_memory_mb") is not None:
            memory_usage.append(sys["peak_memory_mb"])

# === COMPUTE AVERAGES ===
def safe_avg(values):
    return round(statistics.mean(values), 4) if values else 0.0

results = {
    "Total Samples": len(data),
    "Average Wallclock Time (s)": safe_avg(wall_times),
    "Average Prefill TPS": safe_avg(prefill_tps),
    "Average Decode TPS": safe_avg(decode_tps),
    "Average CPU Usage (%)": safe_avg(cpu_usage),
    "Average Peak Memory (MB)": safe_avg(memory_usage),
}

# === DISPLAY RESULTS ===
print("\n--- Consolidated Benchmark Results ---")
for k, v in results.items():
    print(f"{k:<35}: {v}")
