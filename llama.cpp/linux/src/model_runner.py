# src/model_runner.py

import subprocess, time, re, threading, os
from src.system_monitor import SystemMonitor


def run_inference(model_path, prompt, n_predict=128, use_gpu=False, gpu_layers=0):
    """
    Run llama.cpp and parse performance metrics.
    Works with both old and new llama.cpp log formats.
    Supports GPU acceleration via CUDA or Vulkan.
    """

    # Replace with the path to your llama-cli binary
    LLAMA_CPP_BIN = os.path.expanduser("path/to/your/llama-cli/binary")

    cmd = [
        LLAMA_CPP_BIN,
        "-m", model_path,
        "-p", prompt,
        "-n", str(n_predict),
        "-no-cnv"  # suppress color + extra stuff in stdout
    ]

    # Add GPU flags if requested
    if use_gpu:
        if gpu_layers == -1:
            cmd.extend(["-ngl", "999"])  # Offload all layers to GPU (use high number)
        elif gpu_layers > 0:
            cmd.extend(["-ngl", str(gpu_layers)])  # Number of layers to offload to GPU
        # For Vulkan, llama.cpp automatically detects if built with Vulkan support
        # For CUDA, ensure the binary is built with CUDA support

    start_time = time.time()
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Monitor system metrics in background
    monitor = SystemMonitor(process.pid)
    monitor_thread = threading.Thread(target=monitor.run, args=(0.2,), daemon=True)
    monitor_thread.start()

    stdout, stderr = process.communicate()
    end_time = time.time()
    monitor_thread.join()

    # --- Extract timings ---
    prefill_tps, decode_tps, load_time_ms, prompt_eval_time = None, None, None, None
    for line in stderr.splitlines():
        line = line.strip()
        
        if "load time" in line:
            match = re.search(r"load time\s*=\s*([\d.]+)\s*ms", line)
            if match:
                load_time_ms = float(match.group(1))

        # Old format: "prompt eval time = ... ( 40.51 tokens/s)"
        if "prompt eval time" in line:
            match = re.search(r"\(?([\d.]+)\s+tokens\s+per\s+second\)?", line)
            if match:
                prefill_tps = float(match.group(1))
            match_time = re.search(r"prompt eval time\s*=\s*([\d.]+)\s*ms", line)
            if match_time:
                prompt_eval_time = float(match_time.group(1))

    
        if "eval time" in line and "prompt" not in line:
            match = re.search(r"([\d.]+)\s+tokens( per second|/?s(ec)?)", line)
            if match:
                decode_tps = float(match.group(1))

    timings = {
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
        "time_to_first_token_s": load_time_ms + prompt_eval_time if load_time_ms and prompt_eval_time else None,
    }

    system_metrics = monitor.summary()

    return {
        "output": stdout,              # raw model output (no strip)
        "logs": stderr,                # raw llama.cpp logs
        "timings": timings,            # parsed timings dict
        "system_metrics": system_metrics,
        "wallclock_s": end_time - start_time,
    }
