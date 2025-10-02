import subprocess, time, re, threading, os, queue
from src.system_monitor import SystemMonitor


def run_inference(model_path, prompt, n_predict=128, use_gpu=False, gpu_layers=0):
    """
    Run llama.cpp and parse performance metrics.
    Works with both old and new llama.cpp log formats.
    Supports GPU acceleration via CUDA or Vulkan.
    """

    LLAMA_CPP_BIN = r"C:\Users\reach\OneDrive\Desktop\Samsung\on-device-inferencing-and-optimization\llama.cpp\build\bin\Release\llama-cli.exe"
    cmd = [
        LLAMA_CPP_BIN,
        "-m", model_path,
        "-p", prompt,
        "-n", str(n_predict),
        "-no-cnv"
        # Removed --log-disable to allow timing output
    ]

    if use_gpu:
        if gpu_layers == -1:
            cmd.extend(["-ngl", "999"])
        elif gpu_layers > 0:
            cmd.extend(["-ngl", str(gpu_layers)])

    start_time = time.time()
    
    # Use unbuffered I/O
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        stdin=subprocess.PIPE,  # Provide stdin to prevent blocking
        text=True,
        bufsize=0  # Unbuffered
    )
    
    # Close stdin immediately
    process.stdin.close()

    monitor = SystemMonitor(process.pid)
    monitor_thread = threading.Thread(target=monitor.run, args=(0.2,), daemon=True)
    monitor_thread.start()

    # Use queues for thread-safe collection
    stdout_queue = queue.Queue()
    stderr_queue = queue.Queue()
    
    def enqueue_output(pipe, q, name):
        try:
            for line in iter(pipe.readline, ''):
                if line:
                    q.put(line)
        except Exception as e:
            print(f"Error reading {name}: {e}")
        finally:
            pipe.close()
    
    stdout_thread = threading.Thread(
        target=enqueue_output, 
        args=(process.stdout, stdout_queue, "OUT"),
        daemon=True
    )
    stderr_thread = threading.Thread(
        target=enqueue_output, 
        args=(process.stderr, stderr_queue, "ERR"),
        daemon=True
    )
    
    stdout_thread.start()
    stderr_thread.start()
    
    # Wait for process with reasonable timeout
    timeout = 600  # 10 minutes
    try:
        return_code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"ERROR: Process timed out after {timeout}s")
        process.kill()
        try:
            process.wait(timeout=5)
        except:
            process.terminate()
        raise RuntimeError(f"Inference timed out after {timeout} seconds")
    
    end_time = time.time()
    
    # Give threads a moment to finish reading
    time.sleep(0.5)
    
    # Collect all output
    stdout_lines = []
    stderr_lines = []
    
    while not stdout_queue.empty():
        try:
            stdout_lines.append(stdout_queue.get_nowait())
        except queue.Empty:
            break
    
    while not stderr_queue.empty():
        try:
            stderr_lines.append(stderr_queue.get_nowait())
        except queue.Empty:
            break
    
    monitor_thread.join(timeout=2)

    stdout = ''.join(stdout_lines)
    stderr = ''.join(stderr_lines)

    # Extract timings with patterns for llama_perf_context_print format
    prefill_tps, decode_tps, load_time_ms, prompt_eval_time = None, None, None, None
    
    for line in stderr.splitlines():
        line = line.strip()
        
        # Match: llama_perf_context_print: load time = 843.80 ms
        if "load time" in line.lower():
            match = re.search(r"load time\s*=\s*([\d.]+)\s*ms", line, re.IGNORECASE)
            if match:
                load_time_ms = float(match.group(1))

        # Match: llama_perf_context_print: prompt eval time = 301.40 ms / 1323 tokens ( 0.23 ms per token, 4389.47 tokens per second)
        if "prompt eval time" in line.lower():
            # Extract tokens per second (the speed we want)
            match = re.search(r",\s*([\d.]+)\s+tokens\s+per\s+second", line, re.IGNORECASE)
            if match:
                prefill_tps = float(match.group(1))
            # Extract total time in ms
            match_time = re.search(r"prompt eval time\s*=\s*([\d.]+)\s*ms", line, re.IGNORECASE)
            if match_time:
                prompt_eval_time = float(match_time.group(1))
    
        # Match: llama_perf_context_print: eval time = 342.23 ms / 49 runs ( 6.98 ms per token, 143.18 tokens per second)
        if "eval time" in line.lower() and "prompt" not in line.lower():
            # Extract tokens per second
            match = re.search(r",\s*([\d.]+)\s+tokens\s+per\s+second", line, re.IGNORECASE)
            if match:
                decode_tps = float(match.group(1))

    timings = {
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
        "time_to_first_token_s": (load_time_ms + prompt_eval_time) / 1000.0 if (load_time_ms and prompt_eval_time) else None,
        "load_time_ms": load_time_ms,
        "prompt_eval_time_ms": prompt_eval_time
    }

    system_metrics = monitor.summary()

    return {
        "output": stdout,
        "logs": stderr,
        "timings": timings,
        "system_metrics": system_metrics,
        "wallclock_s": end_time - start_time,
    }