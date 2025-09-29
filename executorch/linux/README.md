# ExecuTorch Evaluation Framework (Linux)

This document provides instructions on how to run the evaluation framework for models (in HuggingFace format, exported to .pte) using ExecuTorch for a Linux environment.

## Setup

### 1. Clone the repository and install dependencies

```bash
git clone https://github.com/Arya-Hari/MobiBench.git
cd MobiBench/executorch/linux
```

Dependencies include:

- transformers
- optimum[exporters]
- optimum-executorch
- pandas
- psutil

Follow https://github.com/huggingface/optimum-executorch to install optimum-executorch before executing 

### 2. Export Models to ExecuTorch

Models are automatically exported to `.pte` format on the first run of `main.py` using `optimum-cli`. You just need to provide the HuggingFace `--model_name`.

For example, running with:

```bash
--model_name HuggingFaceTB/SmolLM2-135M-Instruct
```

will:

- Export the model to `.pte` format with the XNNPACK backend.
- Save the tokenizer locally in the same directory.

Exported models are stored under a folder named after the model, e.g.:

```bash
./SmolLM2-135M-Instruct/model.pte
./SmolLM2-135M-Instruct/tokenizer_config.json
```

### 3. Create a model folder (optional)

You may organize models under a `models/` directory if you wish:

```bash
cd ../../
mkdir models
cd models
```

## Run Evaluations

### 1. Run evaluation

To evaluate a model on a dataset from the root directory, use:

```bash
python3 ./executorch/linux/main.py \
    --dataset_type <dataset_type> \
    --csv_path <path_to_dataset> \
    --model_name <huggingface_model_name> \
    --output_json <path_to_output> \
    --n_predict 128
```

Ensure that the `results/` directory exists for output JSONL.

### 2. Available `dataset_type` Options

- `context_qa` → RepLiQA  
- `scientific_mcq` → SciQ  
- `mmlu_mcq` → MMLU  
- `summarization` → CNN/DailyMail  

### 3. Example Usage

#### Context QA (CPU-only):

```bash
python3 ./executorch/linux/main.py \
    --dataset_type context_qa \
    --csv_path ./data/csv/context_qa_dataset.csv \
    --model_name HuggingFaceTB/SmolLM2-135M-Instruct \
    --output_json ./executorch/linux/results/context_qa_smol.jsonl \
    --n_predict 128
```

#### Scientific MCQ:

```bash
python3 ./executorch/linux/main.py \
    --dataset_type scientific_mcq \
    --csv_path ./data/csv/scientific_mcq_dataset.csv \
    --model_name HuggingFaceTB/SmolLM2-135M-Instruct \
    --output_json ./executorch/linux/results/scientific_mcq_smol.jsonl \
    --n_predict 128
```

#### MMLU MCQ:

```bash
python3 ./executorch/linux/main.py \
    --dataset_type mmlu_mcq \
    --csv_path ./data/csv/mmlu_mcq_dataset.csv \
    --model_name HuggingFaceTB/SmolLM2-135M-Instruct \
    --output_json ./executorch/linux/results/mmlu_mcq_smol.jsonl \
    --n_predict 128
```

#### Summarization:

```bash
python3 ./executorch/linux/main.py \
    --dataset_type summarization \
    --csv_path ./data/csv/summarization_dataset.csv \
    --model_name HuggingFaceTB/SmolLM2-135M-Instruct \
    --output_json ./executorch/linux/results/summarization_smol.jsonl \
    --n_predict 128
```

## Notes

- ✅ First run will export model + tokenizer locally to `.pte` format.  
- ✅ Re-runs will reuse the already-exported model and tokenizer.  
- ⚠️ Currently, CPU-only is supported (XNNPACK backend). GPU support may be added later.  

Outputs are stored as JSONL with fields:

- `prompt`  
- `model_output`  
- `logs`  
- `timings`  
- `system_metrics`  
- `wallclock_s`  

### Example Output File

Each line in the JSONL file corresponds to one prompt:

```json
{
  "prompt": "What is the capital of France?",
  "model_output": "The capital of France is Paris.",
  "logs": "Inference successful",
  "timings": {},
  "system_metrics": {"cpu": 12.5, "ram": 2.1},
  "wallclock_s": 0.58
}
```
