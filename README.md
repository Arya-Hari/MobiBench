# MobiBench: Benchmarking Lightweight LLM Inference using llama.cpp on CPU and GPU Platforms

MobiBench is a benchmarking framework for evaluating lightweight, quantized (GGUF) LLMs run with [llama.cpp](https://github.com/ggerganov/llama.cpp) on **consumer laptop CPU and GPU platforms**. It combines system-level measurements (latency, throughput, memory, CPU utilization) with task-level quality metrics (accuracy, BERTScore, LLM-based semantic evaluation) under one runtime, dataset, prompt, and scoring pipeline.

**Scope.** Validation is limited to the consumer laptop CPU/GPU platforms and the model configurations listed below. It does not cover ARM-based smartphones, mobile SoCs/NPUs, single-board computers, or other embedded accelerators. Extending MobiBench to that hardware is future work.

## Models evaluated

| Model | Params | Quantization | CPU | GPU |
|---|---|---|---|---|
| Gemma | 1B | Q4_0 | yes | yes |
| TinyLlama | 1.1B | Q4_0 | yes | yes |
| LLaMA | 3B | Q4_0 | no | yes |
| Phi-2 | 2.7B | Q4_0 | no | yes |
| Phi-2 | 2.7B | Q3_K_M | no | yes |
| Phi-2 | 2.7B | Q2_K | no | yes |

Direct CPU-vs-GPU comparison is available only for Gemma 1B and TinyLlama.

## Hardware

- Gaming laptop: Intel Core i7 (12th gen) with NVIDIA GeForce RTX 3060 (6 GB)
- Laptop: Intel Core i5, no dedicated GPU

## Benchmark configuration

| Setting | Value |
|---|---|
| llama.cpp | build 5970, commit `07a19e27a26f76d34be62da53807f93131fb3cab` |
| CUDA Toolkit / NVIDIA driver | 12.6 / 561.17 |
| Build generator | Ninja |
| GPU offload | `-ngl 999` |
| Generation limit | `-n 128` |
| Conversation mode | `-no-cnv` |
| Context length | 512 tokens |
| Temperature | 0 (greedy) |
| Repetition | One execution per model/dataset/device configuration |

Each sample is run as an independent `llama-cli` invocation.

## Tasks and datasets

| Task | Dataset | Sampling | Size |
|---|---|---|---|
| Contextual QA | [RepLiQA](https://huggingface.co/datasets/ServiceNow/repliqa) | 200 random entries from each of 5 data files | 1,000 |
| Summarization | [CNN/DailyMail](https://huggingface.co/datasets/abisee/cnn_dailymail) | 334 random entries from each of 3 test files | 1,002 |
| Scientific QA | [SciQ](https://huggingface.co/datasets/allenai/sciq) | Official test set | 1,000 |
| Multiple-choice reasoning | [MMLU](https://huggingface.co/datasets/cais/mmlu) | 1,000 random entries from the `all` test split | 1,000 |

## Metrics

**System-level**
- **Time (s):** mean per-sample wall-clock latency, including process start, model loading, prefill, decoding, and teardown. It is not decode-only latency.
- **Prefill TPS / Decode TPS:** tokens per second for prompt processing and generation.
- **CPU (%):** average CPU usage of the inference process only. Values above 100% mean use of multiple logical cores.
- **Mem (MB):** peak host memory (resident set size, via `psutil`) of the inference process.

**Task-level**
- **Summarization:** BERTScore (precision, recall, F1) and an LLM-based semantic score.
- **QA / multiple choice:** accuracy. Prompts are zero-shot with no chat template. Answers are parsed from the generated continuation only, not the echoed prompt. Unparsable responses count as incorrect and stay in the denominator.

**LLM-based evaluation.** `openai/gpt-oss-120b` via the Groq API scores the similarity (0.0 to 1.0) between model and reference summaries on the first 100 valid records, a fixed sequential subset rather than a random sample. Temperature is 0. Seed, top-p, and max output tokens were not set. There are no repeated evaluator runs and no human validation, so scores are descriptive only.

## Repository structure

- `data/`: dataset subsets used in the experiments
- `eval_metrics/`: evaluation and metric scripts
- `llama.cpp/`: llama.cpp used for the benchmark runs
- `results/`: raw outputs and results

## Reproducing the experiments

1. Build llama.cpp at the commit listed above. Use a CUDA build (CUDA 12.6) for GPU runs.
2. Run `llama-cli` once per sample with the settings in the configuration table (`-n 128 -no-cnv`, context 512, temperature 0, and `-ngl 999` for GPU runs).
3. Score the generated outputs with the scripts in `eval_metrics/`.
4. For the LLM-based evaluation, set a `GROQ_API_KEY` environment variable for the Groq API.

## Limitations

- Consumer laptop CPU/GPU only. No mobile SoC, NPU, or embedded results.
- CPU runs exist only for Gemma 1B and TinyLlama.
- One run per configuration, so run-to-run variance is not measured.
- LLM-evaluator settings (seed, top-p) were not controlled or recorded, and there is no human validation.
- llama.cpp is the only runtime evaluated. Other runtimes and model families are future work.

## Citation

If you use this repository in academic work, please cite our paper (preprint forthcoming).
