This doc provides instructions on how to run the evaluation framework for models (in **GGUF** format) using [`llama.cpp`](https://github.com/ggerganov/llama.cpp).  

---

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt

2. **Build llama.cpp**
Build llama.cpp for linux according to the instruction provided in the [official repository](https://github.com/ggerganov/llama.cpp)

3. **Create a model folder**
Create a models directory within the root directory.
    ```bash
    mkdir models
    cd models
Store all models required to be evaluated within this directory.

## Running Test File

1. **Replace paths**
Replace the ```LLAMA_CPP_BIN``` path in [src/test_model_runner.py]() with the path to your llama-cli binary. For example,
    ```bash
    4 LLAMA_CPP_BIN = os.path.expanduser("~/llama.cpp/build/bin/llama-cli")
Replace the model path in [test.py]() with the path to your model in GGUF format. For example,
    ```bash
    4 res = run_inference("~/models/gemma_2b.gguf", "Hi how are you?", 50)

2. **Run the test file**
Run the test file using the command
    ```bash
    python ~/llama.cpp/linux/test.py 
Confirm the outputs.

## Run evaluations

1. **Replace path**
Replace the ```LLAMA_CPP_BIN``` path in [src/model_runner.py]() with the path to your llama-cli binary. For example,
    ```bash
    14 LLAMA_CPP_BIN = os.path.expanduser("~/llama.cpp/build/bin/llama-cli")

2. **Run evaluation**
To evaluate a model on a dataset, use
    ```bash
    python3 ~/llama.cpp/linux/main.py \
    --dataset_type <dataset_type> \
    --csv_path <path_to_dataset> \
    --model_path <path_to_model> \
    --output_json <path_to_output> \
    --n_predict 128

### Available `dataset_type` Options

- **`context_qa`** → [RepLiQA](https://huggingface.co/datasets/ServiceNow/repliqa)  
- **`scientific_mcq`** → [SciQ](https://huggingface.co/datasets/allenai/sciq)  
- **`mmlu_mcq`** → [MMLU](https://huggingface.co/datasets/cais/mmlu)  
- **`summarization`** → [CNN/DailyMail](https://huggingface.co/datasets/abisee/cnn_dailymail)

For example,
    ```bash
    python3 ~/llama.cpp/linux/main.py \
    --dataset_type context_qa \
    --csv_path ~/data/csv/context_qa_dataset.csv \
    --model_path ~/models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
    --output_json ~/llama.cpp/linux/results/context_qa_tinyllama.json \
    --n_predict 128






