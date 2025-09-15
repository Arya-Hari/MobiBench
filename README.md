# MobiBench - Benchmarking LLMs for On-Device Performance

MobiBench is a unified benchmarking suite designed to rigorously evaluate the performance of **large language models (LLMs)** on mobile and edge devices.  
It provides a reproducible framework for assessing **runtime efficiency, accuracy, and resource trade-offs** across multiple execution frameworks, hardware platforms, and natural language tasks.  


## Motivation

With the growing push toward **privacy-preserving, low-latency AI**, deploying LLMs directly on mobile and edge hardware has become crucial. However, existing benchmarks primarily target cloud or server environments and fail to capture the **unique constraints of mobile SoCs** (e.g., limited memory, thermal budgets, and power efficiency).  

MobiBench fills this gap by:
- Comparing **mobile-optimized frameworks** (AI Edge Torch, ExecuTorch, llama.cpp, Bitnet).  
- Evaluating **lightweight models** (TinyLlama, Gemma, LLaMA 1B/3B, Phi family, Bitnet 1B).  
- Testing on **consumer devices** like Google Pixel, Samsung Galaxy, iPhone, and laptops.  
- Measuring both **user-facing metrics** (latency, throughput, accuracy) and **system-level metrics** (memory usage, quantization efficiency, energy consumption).  


## Frameworks Evaluated

- **[AI Edge Torch](https://github.com/facebookresearch/ai-edge-torch)** – PyTorch runtime for mobile and embedded AI.  
- **[ExecuTorch](https://pytorch.org/executorch/)** – Portable, modular runtime extending PyTorch to edge devices.  
- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** – Lightweight, C++ inference engine for quantized LLaMA models.


## Models Benchmarked

- TinyLlama (1B)
- Gemma family  
- LLaMA (1B, 3B)
- Phi (<3B)  
- Bitnet (1B) 

## Hardware Platforms

- Google Pixel (7a/9a)  
- Samsung Galaxy S24 (tentative)  
- iPhone 
- MacBook  
- Laptop (Intel Core i5)  


## Tasks and Datasets

- **Summarization** → CNN/Daily Mail  
- **General QA** → MMLU, ReplicQA, SciQ  

These tasks capture reasoning, factual recall, and scientific knowledge.  


## Metrics

MobiBench evaluates models along two axes:

- **Application-Level Metrics**  
  - Prefill and decode speed (tokens/sec)  
  - Time-to-first-token (sec)  
  - Accuracy (task-specific)

- **System-Level Metrics**  
  - Peak memory usage  
  - Quantization efficiency  
  - Energy consumption
    
## Experimental Setup

- Each framework–device–task combination is run under **controlled conditions**.  
- Identical models are tested across different runtimes to isolate framework overhead.  
- Performance is averaged over multiple runs for reproducibility.  

For more detailed information on how to reporduce the benchmark, go to docs.

## Goals

- Provide a **standardized evaluation suite** for on-device LLMs.  
- Enable researchers and developers to identify **framework–hardware trade-offs**.  
- Lay the foundation for **future model/runtime/hardware co-design** in mobile AI.  

## References

Key works informing this project include:  
- Han et al., *Deep Compression* (ICLR 2016)  
- Frantar et al., *GPTQ* (NeurIPS 2022)  
- Lin et al., *AWQ* (2023)  
- AI EdgeTorch (Meta AI, 2023)  
- PyTorch Team, *ExecuTorch* (2023)  
- Xu et al., *EdgeFormer* (ICML 2023)  

(See the full paper for an extended reference list.)  

## Citation
If you use this repository in academic work, please cite our paper (preprint coming soon).
