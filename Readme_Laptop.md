# MobiBench Inference on Laptop GPU (Windows)

This guide provides plug-and-play commands to rebuild llama.cpp with CUDA, run all inferences on your Nvidia GPU, and generate results.

## Step 1: Rebuild llama.cpp with CUDA Support
Your GPU was idle because the build lacked CUDA. Rebuild it:

1. Open **"x64 Native Tools Command Prompt for VS 2022"** (from Start menu).
2. Run these commands in sequence:

```
cd C:\Users\reach\OneDrive\Desktop\Samsung\on-device-inferencing-and-optimization\llama.cpp
rd /s /q build-x64-windows-msvc-release  (remove old build if exists)
cmake -B build-cuda -G "Visual Studio 17 2022" -A x64 -DGGML_CUDA=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF
cmake --build build-cuda --config Release --target llama-cli
```

The binary will be at: `C:\Users\reach\OneDrive\Desktop\Samsung\on-device-inferencing-and-optimization\llama.cpp\build-cuda\bin\Release\llama-cli.exe`

(If CMake fails, ensure CUDA Toolkit is installed and environment variables set.)

## Step 2: Update Script Path
Edit `MobiBench\llama.cpp\windows\model_runner.py`, change line 11 to:
```
LLAMA_CPP_BIN = r"C:\Users\reach\OneDrive\Desktop\Samsung\on-device-inferencing-and-optimization\llama.cpp\build-cuda\bin\Release\llama-cli.exe"
```

## Step 3: Models Used
All available small models in `models/` folder:
- gemma-2b.Q4_0.gguf
- llama-3.2-1b.q4_0.gguf
- phi-2.Q4_0.gguf
- tinyllama-1.1b-chat-v1.0.Q4_0.gguf

Path prefix: `../../../models/`

## Step 4: Run All Inferences (Plug-and-Play Commands)
Open PowerShell/CMD in `MobiBench\llama.cpp\windows\` (activate venv if needed).

Results will be stored in `../../../MobiBench/results/llama.cpp/windows/gpu/` (creates folder automatically).

Each command runs ~1000 samples on GPU (layers -1 = all to GPU). Track GPU usage in Task Manager - it should show ~1-2GB VRAM usage.

### Gemma-2B
```
python main.py --dataset_type context_qa --csv_path ../../../MobiBench/data/csv/context_qa_dataset.csv --model_path ../../../models/gemma-2b.Q4_0.gguf --output_json ../../../MobiBench/results/llama.cpp/windows/gpu/gemma_context_qa.jsonl --use_gpu --gpu_layers -1 --n_predict 128

python main.py --dataset_type scientific_mcq --csv_path ../../../MobiBench/data/csv/science_qa_dataset.csv --model_path ../../../models/gemma-2b.Q4_0.gguf --output_json ../../../MobiBench/results/llama.cpp/windows/gpu/gemma_science_mcq.jsonl --use_gpu --gpu_layers -1 --n_predict 128

python main.py --dataset_type mmlu_mcq --csv_path ../../../MobiBench/data/csv/mmlu_dataset.csv --model_path ../../../models/gemma-2b.Q4_0.gguf --output_json ../../../MobiBench/results/llama.cpp/windows/gpu/gemma_mmlu_mcq.jsonl --use_gpu --gpu_layers -1 --n_predict 128

python main.py --dataset_type summarization --csv_path ../../../MobiBench/data/csv/summarization_dataset.csv --model_path ../../../models/gemma-2b.Q4_0.gguf --output_json ../../../MobiBench/results/llama.cpp/windows/gpu/gemma_summarization.jsonl --use_gpu --gpu_layers -1 --n_predict 128
```

### LLaMA-3.2-1B
```
python main.py --dataset_type context_qa --csv_path ../../../MobiBench/data/csv/context_qa_dataset.csv --model_path ../../../models/llama-3.2-1b.q4_0.gguf --output_json ../../../MobiBench/results/llama.cpp/windows/gpu/llama32_context_qa.jsonl --use_gpu --gpu_layers -1 --n_predict 128

python main.py --dataset_type scientific_mcq --csv_path ../../../MobiBench/data/csv/science_qa_dataset.csv --model_path ../../../models/llama-3.2-1b.q4_0.gguf --output_json ../../../MobiBench/results/llama.cpp/windows/gpu/llama32_science_mcq.jsonl --use_gpu --gpu_layers -1 --n_predict 128

python main.py --dataset_type mmlu_mcq --csv_path ../../../MobiBench/data/csv/mmlu_dataset.csv --model_path ../../../models/llama-3.2-1b.q4_0.gguf --output_json ../../../MobiBench/results/llama.cpp/windows/gpu/llama32_mmlu_mcq.jsonl --use_gpu --gpu_layers -1 --n_predict 128

python main.py --dataset_type summarization --csv_path ../../../MobiBench/data/csv/summarization_dataset.csv --model_path ../../../models/llama-3.2-1b.q4_0.gguf --output_json ../../../MobiBench/results/llama.cpp/windows/gpu/llama32_summarization.jsonl --use_gpu --gpu_layers -1 --n_predict 128
```

### Phi-2
```
python main.py --dataset_type context_qa --csv_path ../../../MobiBench/data/csv/context_qa_dataset.csv --model_path ../../../models/phi-2.Q4_0.gguf --output_json ../../../MobiBench/results/llama.cpp/windows/gpu/phi2_context_qa.jsonl --use_gpu --gpu_layers -1 --n_predict 128

python main.py --dataset_type scientific_mcq --csv_path ../../../MobiBench/data/csv/science_qa_dataset.csv --model_path ../../../models/phi-2.Q4_0.gguf --output_json ../../../MobiBench/results/llama.cpp/windows/gpu/phi2_science_mcq.jsonl --use_gpu --gpu_layers -1 --n_predict 128

python main.py --dataset_type mmlu_mcq --csv_path ../../../MobiBench/data/csv/mmlu_dataset.csv --model_path ../../../models/phi-2.Q4_0.gguf --output_json ../../../MobiBench/results/llama.cpp/windows/gpu/phi2_mmlu_mcq.jsonl --use_gpu --gpu_layers -1 --n_predict 128

python main.py --dataset_type summarization --csv_path ../../../MobiBench/data/csv/summarization_dataset.csv --model_path ../../../models/phi-2.Q4_0.gguf --output_json ../../../MobiBench/results/llama.cpp/windows/gpu/phi2_summarization.jsonl --use_gpu --gpu_layers -1 --n_predict 128
```

### TinyLLaMA-1.1B
```
python main.py --dataset_type context_qa --csv_path ../../../MobiBench/data/csv/context_qa_dataset.csv --model_path ../../../models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf --output_json ../../../MobiBench/results/llama.cpp/windows/gpu/tinyllama_context_qa.jsonl --use_gpu --gpu_layers -1 --n_predict 128

python main.py --dataset_type scientific_mcq --csv_path ../../../MobiBench/data/csv/science_qa_dataset.csv --model_path ../../../models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf --output_json ../../../MobiBench/results/llama.cpp/windows/gpu/tinyllama_science_mcq.jsonl --use_gpu --gpu_layers -1 --n_predict 128

python main.py --dataset_type mmlu_mcq --csv_path ../../../MobiBench/data/csv/mmlu_dataset.csv --model_path ../../../models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf --output_json ../../../MobiBench/results/llama.cpp/windows/gpu/tinyllama_mmlu_mcq.jsonl --use_gpu --gpu_layers -1 --n_predict 128

python main.py --dataset_type summarization --csv_path ../../../MobiBench/data/csv/summarization_dataset.csv --model_path ../../../models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf --output_json ../../../MobiBench/results/llama.cpp/windows/gpu/tinyllama_summarization.jsonl --use_gpu --gpu_layers -1 --n_predict 128
```

## Step 5: Analyze Results
After runs, analyze each .jsonl file in the gpu folder for averages.

Create a script `analyze_results.py` in `MobiBench/results/llama.cpp/windows/gpu/` with code from earlier response.

Run: `python analyze_results.py`

Output summaries like:
```
gemma_context_qa: Prefill TPS: 123.45, Decode TPS: 67.89, Wall: 1.23s, GPU Mem: 1024MB, Samples: 1000
```

## Step 6: Update README
Use the analysis output to update this file's results section, and update main `MobiBench/README.md`.

## File Locations
- Scripts: `MobiBench\llama.cpp\windows\`
- Datasets: `MobiBench\data\csv\`
- Models: `root\models\`
- Results: `MobiBench\results\llama.cpp\windows\gpu\`
- This guide: `MobiBench\Readme_Laptop.md`
