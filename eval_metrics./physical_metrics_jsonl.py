import json
import statistics

path = "/Users/rohitsuresh/Downloads/honeypot/phi2_science_mcq.jsonl"

wall_times = []
prefill_tps = []
decode_tps = []
cpu_usage = []
memory_usage = []

with open(path, "r") as f:
    for line in f:
        if not line.strip():
            continue
        entry = json.loads(line)

        wall_times.append(entry.get("wallclock_s", 0))

        timings = entry.get("timings", {})
        if timings.get("prefill_tps"): prefill_tps.append(timings["prefill_tps"])
        if timings.get("decode_tps"): decode_tps.append(timings["decode_tps"])

        sys = entry.get("system_metrics", {})
        if sys.get("avg_cpu_percent"): cpu_usage.append(sys["avg_cpu_percent"])
        if sys.get("peak_memory_mb"): memory_usage.append(sys["peak_memory_mb"])

def safe_avg(values):
    return round(statistics.mean(values), 4) if values else 0.0

results = {
    "Total Samples": len(wall_times),
    "Average Wallclock Time (s)": safe_avg(wall_times),
    "Average Prefill TPS": safe_avg(prefill_tps),
    "Average Decode TPS": safe_avg(decode_tps),
    "Average CPU Usage (%)": safe_avg(cpu_usage),
    "Average Peak Memory (MB)": safe_avg(memory_usage),
}

print("\n--- Consolidated Benchmark Results ---")
for k, v in results.items():
    print(f"{k:<35}: {v}")
