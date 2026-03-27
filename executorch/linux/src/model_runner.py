import os
import time
import threading
import torch
from transformers import AutoTokenizer
from optimum.executorch import ExecuTorchModelForCausalLM
from src.system_monitor import SystemMonitor

def run_inference(model_path, prompt, n_predict=128, use_gpu=False, gpu_layers=0):
    """
    Run ExecuTorch inference and gather performance metrics.
    This version uses the reliable `text_generation` method.
    """
    logs = []
    start_wall_time = time.time()

    try:
        logs.append("Loading ExecuTorch model...")
        model = ExecuTorchModelForCausalLM.from_pretrained(model_path)
        logs.append("Model loaded successfully.")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            logs.append("Local tokenizer loaded.")
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
            logs.append("Remote tokenizer loaded: HuggingFaceTB/SmolLM2-135M-Instruct")

    except Exception as e:
        return {
            "output": None, "logs": "\n".join(logs) + f"\nError loading model/tokenizer: {e}",
            "timings": {}, "system_metrics": {}, "wallclock_s": time.time() - start_wall_time,
        }

    pid = os.getpid()
    monitor = SystemMonitor(pid)
    monitor_thread = threading.Thread(target=monitor.run, args=(0.2,), daemon=True)
    monitor_thread.start()

    num_prompt_tokens = len(tokenizer.encode(prompt))
    final_output = None
    overall_tps = None
    
    try:
        logs.append(f"Starting inference for ~{n_predict} tokens (using text_generation)...")
        
        # Calculate the total sequence length
        max_len = num_prompt_tokens + n_predict
        
        inference_start_time = time.time()
        
        # *** CORE CHANGE: Revert to the stable, high-level text_generation method ***
        generated_text = model.text_generation(
            tokenizer=tokenizer,
            prompt=prompt,
            max_seq_len=max_len
        )
        
        inference_end_time = time.time()
        total_inference_time = inference_end_time - inference_start_time
        
        # The output of text_generation is only the newly generated part
        final_output = generated_text
        
        # To get TPS, we must tokenize the output string to count the tokens
        num_generated_tokens = len(tokenizer.encode(final_output))
        
        if total_inference_time > 0 and num_generated_tokens > 0:
            overall_tps = num_generated_tokens / total_inference_time
            
        logs.append(f"Inference successful. Generated {num_generated_tokens} tokens in {total_inference_time:.2f} seconds.")

    except Exception as e:
        logs.append(f"Error during inference: {e}")

    end_wall_time = time.time()

    monitor.stop()
    monitor_thread.join(timeout=1)

    timings = {
        "prefill_tps": None, "decode_tps": None, "time_to_first_token_s": None,
        "overall_tps": overall_tps
    }
    system_metrics = monitor.summary()

    return {
        "output": f"{prompt}{final_output}" if final_output is not None else prompt,
        "logs": "\n".join(logs),
        "timings": timings,
        "system_metrics": system_metrics,
        "wallclock_s": end_wall_time - start_wall_time,
    }