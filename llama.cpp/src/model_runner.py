# src/inference/run_llama_cpp.py

import subprocess, time, re, threading
from src.system_monitor import SystemMonitor

def run_inference(model_path, prompt, n_predict=128):
    """
    Runs llama.cpp main binary with subprocess.
    Parses timing info from stderr and captures system metrics.
    """
    cmd = [
        "./main",
        "-m", model_path,
        "-p", prompt,
        "-n", str(n_predict),
        "--log-disable"
    ]

    start_time = time.time()
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Monitor system metrics
    monitor = SystemMonitor(process.pid)
    monitor_thread = threading.Thread(target=monitor.run, args=(0.2,), daemon=True)
    monitor_thread.start()

    stdout, stderr = process.communicate()
    end_time = time.time()
    monitor_thread.join()

    # --- Extract timings ---
    prefill_tps, decode_tps, ttft = None, None, None
    for line in stderr.splitlines():
        # Example: "prompt eval time =  1234.56 ms / 50 tokens (  40.51 tokens/s)"
        if "prompt eval time" in line:
            match = re.search(r"\(([\d.]+)\s+tokens/s\)", line)
            if match:
                prefill_tps = float(match.group(1))

        # Example: "eval time =  5678.90 ms / 100 tokens (  17.61 tokens/s)"
        if re.match(r"eval time", line.strip()):
            match = re.search(r"\(([\d.]+)\s+tokens/s\)", line)
            if match:
                decode_tps = float(match.group(1))

        # Example: "time to first token =  250.12 ms"
        if "time to first token" in line:
            match = re.search(r"([\d.]+)\s*ms", line)
            if match:
                ttft = float(match.group(1)) / 1000.0  # convert to seconds

    timings = {
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
        "time_to_first_token_s": ttft,
    }

    system_metrics = monitor.summary()

    return {
        "output": stdout.strip(),
        "timings": timings,
        "system_metrics": system_metrics,
        "wallclock_s": end_time - start_time
    }
