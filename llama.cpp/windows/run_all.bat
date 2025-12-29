@echo off
REM Activate virtual environment if needed
REM call path\to\venv\Scripts\activate.bat

REM ------------------- Gemma-2B -------------------
REM python main.py --dataset_type context_qa --csv_path ../../../MobiBench/data/csv/context_qa_dataset.csv --model_path ../../../models/gemma-2b.Q4_0.gguf --output_json ../../../MobiBench/results/llama.cpp/windows/gpu/gemma_context_qa.jsonl --use_gpu --gpu_layers -1 --n_predict 128
REM python main.py --dataset_type scientific_mcq --csv_path ../../../MobiBench/data/csv/science_qa_dataset.csv --model_path ../../../models/gemma-2b.Q4_0.gguf --output_json ../../../MobiBench/results/llama.cpp/windows/gpu/gemma_science_mcq.jsonl --use_gpu --gpu_layers -1 --n_predict 128
REM python main.py --dataset_type mmlu_mcq --csv_path ../../../MobiBench/data/csv/mmlu_dataset.csv --model_path ../../../models/gemma-2b.Q4_0.gguf --output_json ../../../MobiBench/results/llama.cpp/windows/gpu/gemma_mmlu_mcq.jsonl --use_gpu --gpu_layers -1 --n_predict 128
REM python main.py --dataset_type summarization --csv_path ../../../MobiBench/data/csv/summarization_dataset.csv --model_path ../../../models/gemma-2b.Q4_0.gguf --output_json ../../../MobiBench/results/llama.cpp/windows/gpu/gemma_summarization.jsonl --use_gpu --gpu_layers -1 --n_predict 128

REM ------------------- LLaMA-3.2-1B -------------------
REM python main.py --dataset_type context_qa --csv_path ../../../MobiBench/data/csv/context_qa_dataset.csv --model_path ../../../models/llama-3.2-1b.q4_0.gguf --output_json ../../../MobiBench/results/llama.cpp/windows/gpu/llama32_context_qa.jsonl --use_gpu --gpu_layers -1 --n_predict 128
REM python main.py --dataset_type scientific_mcq --csv_path ../../../MobiBench/data/csv/science_qa_dataset.csv --model_path ../../../models/llama-3.2-1b.q4_0.gguf --output_json ../../../MobiBench/results/llama.cpp/windows/gpu/llama32_science_mcq.jsonl --use_gpu --gpu_layers -1 --n_predict 128
REM python main.py --dataset_type mmlu_mcq --csv_path ../../../MobiBench/data/csv/mmlu_dataset.csv --model_path ../../../models/llama-3.2-1b.q4_0.gguf --output_json ../../../MobiBench/results/llama.cpp/windows/gpu/llama32_mmlu_mcq.jsonl --use_gpu --gpu_layers -1 --n_predict 128
REM python main.py --dataset_type summarization --csv_path ../../../MobiBench/data/csv/summarization_dataset.csv --model_path ../../../models/llama-3.2-1b.q4_0.gguf --output_json ../../../MobiBench/results/llama.cpp/windows/gpu/llama32_summarization.jsonl --use_gpu --gpu_layers -1 --n_predict 128

REM ------------------- Phi-2 -------------------
REM python main.py --dataset_type context_qa --csv_path ../../../MobiBench/data/csv/context_qa_dataset.csv --model_path ../../../models/phi-2.Q4_0.gguf --output_json ../../../MobiBench/results/llama.cpp/windows/gpu/phi2_context_qa.jsonl --use_gpu --gpu_layers -1 --n_predict 128
REM python main.py --dataset_type scientific_mcq --csv_path ../../../MobiBench/data/csv/science_qa_dataset.csv --model_path ../../../models/phi-2.Q4_0.gguf --output_json ../../../MobiBench/results/llama.cpp/windows/gpu/phi2_science_mcq.jsonl --use_gpu --gpu_layers -1 --n_predict 128
REM python main.py --dataset_type mmlu_mcq --csv_path ../../../MobiBench/data/csv/mmlu_dataset.csv --model_path ../../../models/phi-2.Q4_0.gguf --output_json ../../../MobiBench/results/llama.cpp/windows/gpu/phi2_mmlu_mcq.jsonl --use_gpu --gpu_layers -1 --n_predict 128
REM python main.py --dataset_type summarization --csv_path ../../../MobiBench/data/csv/summarization_dataset.csv --model_path ../../../models/phi-2.Q4_0.gguf --output_json ../../../MobiBench/results/llama.cpp/windows/gpu/phi2_summarization.jsonl --use_gpu --gpu_layers -1 --n_predict 128

REM ------------------- Gemma-1B -------------------
REM python main.py --dataset_type context_qa --csv_path ../../../MobiBench/data/csv/context_qa_dataset.csv --model_path ../../../models/gemma-3-1b-it-Q4_0.gguf --output_json ../../../MobiBench/results/llama.cpp/windows/gpu/gemma-1b_context_qa.jsonl --use_gpu --gpu_layers -1 --n_predict 128
REM python main.py --dataset_type scientific_mcq --csv_path ../../../MobiBench/data/csv/science_qa_dataset.csv --model_path ../../../models/gemma-3-1b-it-Q4_0.gguf --output_json ../../../MobiBench/results/llama.cpp/windows/gpu/gemma-1b_science_mcq.jsonl --use_gpu --gpu_layers -1 --n_predict 128
REM python main.py --dataset_type mmlu_mcq --csv_path ../../../MobiBench/data/csv/mmlu_dataset.csv --model_path ../../../models/gemma-3-1b-it-Q4_0.gguf --output_json ../../../MobiBench/results/llama.cpp/windows/gpu/gemma-1b_mmlu_mcq.jsonl --use_gpu --gpu_layers -1 --n_predict 128
REM python main.py --dataset_type summarization --csv_path ../../../MobiBench/data/csv/summarization_dataset.csv --model_path ../../../models/gemma-3-1b-it-Q4_0.gguf --output_json ../../../MobiBench/results/llama.cpp/windows/gpu/gemma-1b_summarization.jsonl --use_gpu --gpu_layers -1 --n_predict 128

REM ------------------- TinyLLaMA-1.1B -------------------
REM python main.py --dataset_type context_qa --csv_path ../../../MobiBench/data/csv/context_qa_dataset.csv --model_path ../../../models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf --output_json ../../../MobiBench/results/llama.cpp/windows/gpu/tinyllama_context_qa.jsonl --use_gpu --gpu_layers -1 --n_predict 128
REM python main.py --dataset_type scientific_mcq --csv_path ../../../MobiBench/data/csv/science_qa_dataset.csv --model_path ../../../models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf --output_json ../../../MobiBench/results/llama.cpp/windows/gpu/tinyllama_science_mcq.jsonl --use_gpu --gpu_layers -1 --n_predict 128
python main.py --dataset_type mmlu_mcq --csv_path ../../../MobiBench/data/csv/mmlu_dataset.csv --model_path ../../../models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf --output_json ../../../MobiBench/results/llama.cpp/windows/gpu/tinyllama_mmlu_mcq.jsonl --use_gpu --gpu_layers -1 --n_predict 128
python main.py --dataset_type summarization --csv_path ../../../MobiBench/data/csv/summarization_dataset.csv --model_path ../../../models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf --output_json ../../../MobiBench/results/llama.cpp/windows/gpu/tinyllama_summarization.jsonl --use_gpu --gpu_layers -1 --n_predict 128


echo All inference runs completed!
pause
