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

```bash
python3 ./llama.cpp/linux/main.py \
    --dataset_type context_qa \
    --csv_path ~/data/csv/context_qa_dataset.csv \
    --model_path ~/models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
    --output_json ~/llama.cpp/linux/results/context_qa_tinyllama.json \
    --n_predict 128
```
