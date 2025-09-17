import subprocess, os

# replace with the path to the llama-cli binary
LLAMA_CPP_BIN = os.path.expanduser("your-path-to-llama-cpp-llama-cli")

def run_raw(model_path, prompt, n_predict=128):
    cmd = [
        LLAMA_CPP_BIN,
        "-m", model_path,
        "-p", prompt,
        "-n", str(n_predict),
        "-no-cnv",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return result.stdout, result.stderr
