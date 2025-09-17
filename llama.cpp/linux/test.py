from src.model_runner import run_inference

# replace with the model path
res = run_inference("your/model/path", "Hi how are you?", 50)

print("=== RAW OUTPUT ===")
print(res["output"])
print("\n=== RAW LOGS ===")
print(res["logs"])
print("\n=== PARSED TIMINGS ===")
print(res["timings"])
print("\n=== SYSTEM METRICS ===")
print(res["system_metrics"])
print("\n=== WALLCLOCK TIME (s) ===")
print(res["wallclock_s"])
