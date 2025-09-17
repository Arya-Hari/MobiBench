# main.py

import argparse
import json
from src.data_loader import get_loader
from src.model_runner import run_inference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", choices=["context_qa", "mcq", "summarization"], required=True)
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_json", default="results/output.json")
    parser.add_argument("--n_predict", type=int, default=128)
    parser.add_argument("--use_gpu", action="store_true", help="Enable GPU acceleration")
    parser.add_argument("--gpu_layers", type=int, default=0, help="Number of layers to offload to GPU (0 = auto, -1 = all layers)")
    args = parser.parse_args()

    # Load dataset
    loader = get_loader(args.dataset_type, args.csv_path)
    prompts = loader.build_prompts()

    results = []
    for i, entry in enumerate(prompts, start=1):
        print(f"Processing {i}/{len(prompts)}")
        result = run_inference(args.model_path, entry["prompt"], args.n_predict, args.use_gpu, args.gpu_layers)

        # Store model outputs + metrics
        entry["model_output"] = result["output"]
        entry["logs"] = result["logs"]                  # raw llama.cpp logs
        entry["timings"] = result["timings"]            # prefill/decode TPS, latency
        entry["system_metrics"] = result["system_metrics"]  # CPU/mem stats
        entry["wallclock_s"] = result["wallclock_s"]    # wallclock time

        results.append(entry)

    # Save results to JSON
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {args.output_json}")


if __name__ == "__main__":
    main()
