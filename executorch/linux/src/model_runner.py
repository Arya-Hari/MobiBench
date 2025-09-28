import os
import time
import threading
from transformers import AutoTokenizer
from optimum.executorch import ExecuTorchModelForCausalLM
from src.system_monitor import SystemMonitor


def run_inference(model_path, prompt, n_predict=128, use_gpu=False, gpu_layers=0):
    """
    Run ExecuTorch inference and gather performance metrics.
    """
    try:
        # Load ExecuTorch model from exported directory
        model = ExecuTorchModelForCausalLM.from_pretrained(model_path)

        # Try local tokenizer first (offline use)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception:
            print("⚠️ Tokenizer not found locally, falling back to HuggingFace repo...")
            tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

    except Exception as e:
        return {
            "output": None,
            "logs": f"Error loading model/tokenizer: {e}",
            "timings": {},
            "system_metrics": {},
            "wallclock_s": 0,
        }

    # System monitor setup
    pid = os.getpid()
    monitor = SystemMonitor(pid)
    monitor_thread = threading.Thread(target=monitor.run, args=(0.2,), daemon=True)
    monitor_thread.start()

    start_time = time.time()

    # Inference
    try:
        generated_text = model.text_generation(
            tokenizer=tokenizer,
            prompt=prompt,
            max_seq_len=n_predict
        )
        output = generated_text
        logs = "Inference successful"
    except Exception as e:
        output = None
        logs = f"Error during inference: {e}"

    end_time = time.time()

    # Stop monitor thread safely
    monitor.stop()
    monitor_thread.join(timeout=1)

    timings = {
        "prefill_tps": None,
        "decode_tps": None,
        "time_to_first_token_s": None,
    }

    system_metrics = monitor.summary()

    return {
        "output": output,
        "logs": logs,
        "timings": timings,
        "system_metrics": system_metrics,
        "wallclock_s": end_time - start_time,
    }
