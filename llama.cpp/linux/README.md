This document provides instructions on how to run the evaluation framework for models (in **GGUF** format) using [`llama.cpp`](https://github.com/ggerganov/llama.cpp) for a Linux environment.

## Setup

### 1. Clone the repository and install dependencies
```bash
git clone https://github.com/Arya-Hari/MobiBench.git
cd MobiBench/llama.cpp/linux
pip install -r requirements.txt
```

### 2. Build llama.cpp
Build llama.cpp for Linux according to the instruction provided in the [official repository](https://github.com/ggerganov/llama.cpp).

#### For GPU Support (CUDA/Vulkan)
To enable GPU acceleration, you need to build llama.cpp with GPU support:

**For CUDA (NVIDIA GPUs):**
```bash
# Install CUDA toolkit first (see NVIDIA documentation)
cd llama.cpp
mkdir build
cd build
cmake .. -DLLAMA_CUDA=ON
cmake --build . --config Release
```

**For Vulkan (Cross-platform GPU support):**
```bash
cd llama.cpp
mkdir build
cd build
cmake .. -DLLAMA_VULKAN=ON
cmake --build . --config Release
```

**For WSL with NVIDIA GPU:**
To enable CUDA support in WSL, follow these steps:

1. **Install NVIDIA Drivers on Windows:**
   - Download and install the latest NVIDIA drivers for your GPU from https://www.nvidia.com/Download/index.aspx
   - Ensure you have a CUDA-capable NVIDIA GPU

2. **Verify CUDA Installation on Windows:**
   ```cmd
   # Open Command Prompt or PowerShell and run:
   nvidia-smi

   # This should show your GPU info and CUDA version
   # Look for "CUDA Version: X.Y" in the output
   ```

   **Alternative verification methods:**
   - Open NVIDIA Control Panel → System Information → Components → check NVCUDA.DLL version
   - Check Windows Settings → Apps → look for "NVIDIA CUDA Toolkit" or "CUDA" entries
   - Open Device Manager → Display adapters → right-click NVIDIA GPU → Properties → Driver tab

3. **Enable WSL2:**
   - WSL2 is required for GPU support. If you have WSL1, upgrade to WSL2:
   ```powershell
   wsl --set-version <distribution-name> 2
   ```

4. **Install CUDA Toolkit in WSL:**
   ```bash
   # Update package lists
   sudo apt update

   # Install CUDA toolkit (choose the version that matches your Windows drivers)
   wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt-get update
   sudo apt-get install cuda-toolkit-12-2  # or latest version

   # Add CUDA to PATH (add to ~/.bashrc)
   echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc
   ```

4. **Verify Installation:**
   ```bash
   nvidia-smi
   nvcc --version
   ```

5. **Build llama.cpp with CUDA:**
   ```bash
   cd llama.cpp
   mkdir build
   cd build
   cmake .. -DLLAMA_CUDA=ON
   cmake --build . --config Release
   ```

**Troubleshooting:**
- If `nvidia-smi` doesn't work, ensure NVIDIA drivers are installed on Windows
- If CUDA installation fails, check that you're using WSL2
- For driver version compatibility, visit: https://docs.nvidia.com/cuda/wsl-user-guide/index.html

### 3. Create a model folder
Create a models directory within the root directory and store all models required to be evaluated within this directory:

```bash
cd ../..
mkdir models
cd models
```

## Running Test File

### 1. Replace paths
Replace the `LLAMA_CPP_BIN` path in `llama.cpp/linux/src/model_runner.py` with the path to your llama-cli binary. For example:

```python
LLAMA_CPP_BIN = os.path.expanduser("~/llama.cpp/build/bin/llama-cli")
```

Replace the model path in `llama.cpp/linux/test.py` with the path to your model in GGUF format. For example:

```python
res = run_inference("./models/gemma_2b.gguf", "Hi how are you?", 50)
```

### 2. Run the test file
From the main directory, run the test file using the command:

```bash
python ./llama.cpp/linux/test.py
```

Confirm the outputs. Go back to the root directory.

## Run Evaluations

### 1. Replace path
Replace the `LLAMA_CPP_BIN` path in `llama.cpp/linux/src/model_runner.py` with the path to your llama-cli binary. For example:

```python
LLAMA_CPP_BIN = os.path.expanduser("~/llama.cpp/build/bin/llama-cli")
```

### 2. Run evaluation
To evaluate a model on a dataset from the root directory, use:

```bash
python3 ./llama.cpp/linux/main.py \
    --dataset_type <dataset_type> \
    --csv_path <path_to_dataset> \
    --model_path <path_to_model> \
    --output_json <path_to_output> \
    --n_predict 128
```

#### Available `dataset_type` Options

- **`context_qa`** → [RepLiQA](https://huggingface.co/datasets/ServiceNow/repliqa)
- **`scientific_mcq`** → [SciQ](https://huggingface.co/datasets/allenai/sciq)
- **`mmlu_mcq`** → [MMLU](https://huggingface.co/datasets/cais/mmlu)
- **`summarization`** → [CNN/DailyMail](https://huggingface.co/datasets/abisee/cnn_dailymail)

#### Example Usage

**CPU-only evaluation:**
```bash
python3 ./llama.cpp/linux/main.py \
    --dataset_type context_qa \
    --csv_path ./data/csv/context_qa_dataset.csv \
    --model_path ./models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
    --output_json ./llama.cpp/linux/results/context_qa_tinyllama.json \
    --n_predict 128
```

**GPU-accelerated evaluation:**
```bash
python3 ./llama.cpp/linux/main.py \
    --dataset_type context_qa \
    --csv_path ./data/csv/context_qa_dataset.csv \
    --model_path ./models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
    --output_json ./llama.cpp/linux/results/context_qa_tinyllama_gpu.json \
    --n_predict 128 \
    --use_gpu \
    --gpu_layers 20  # Offload 20 layers to GPU, adjust based on your GPU memory
```

**Full GPU offload:**
```bash
python3 ./llama.cpp/linux/main.py \
    --dataset_type context_qa \
    --csv_path ./data/csv/context_qa_dataset.csv \
    --model_path ./models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
    --output_json ./llama.cpp/linux/results/context_qa_tinyllama_gpu_full.json \
    --n_predict 128 \
    --use_gpu \
    --gpu_layers -1  # Offload all layers to GPU
```
