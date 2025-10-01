import argparse
import json
import os
from src.data_loader import get_loader
from model_runner import run_inference  # Import from current directory

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", choices=["context_qa", "scientific_mcq", "mmlu_mcq", "summarization"], required=True)
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_json", default="results/output.jsonl")
    parser.add_argument("--n_predict", type=int, default=128)
    parser.add_argument("--use_gpu", action="store_true", help="Enable GPU acceleration")
    parser.add_argument("--gpu_layers", type=int, default=0, help="Number of layers to offload to GPU (0 = auto, -1 = all layers)")
    args = parser.parse_args()

    # Load dataset
    loader = get_loader(args.dataset_type, args.csv_path)
    prompts = loader.build_prompts()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    for i, entry in enumerate(prompts, start=1):
        print(f"Processing {i}/{len(prompts)}")
        result = run_inference(args.model_path, entry["prompt"], args.n_predict, args.use_gpu, args.gpu_layers)

        # Store model outputs + metrics
        entry["model_output"] = result["output"]
        entry["logs"] = result["logs"]
        entry["timings"] = result["timings"]
        entry["system_metrics"] = result["system_metrics"]
        entry["wallclock_s"] = result["wallclock_s"]

        # Append result as one line of JSON
        with open(args.output_json, "a", encoding='utf-8') as f:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

        print(f"Checkpoint saved ({i}/{len(prompts)})")

    print(f"All results written to {args.output_json}")

if __name__ == "__main__":
    main()
