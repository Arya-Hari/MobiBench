# main.py

import argparse
import json
import os
from src.data_loader import get_loader
from src.model_runner import run_inference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", choices=["context_qa", "mcq", "summarization"], required=True)
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_json", default="results/output.jsonl")
    parser.add_argument("--n_predict", type=int, default=128)
    args = parser.parse_args()

    # Load dataset
    loader = get_loader(args.dataset_type, args.csv_path)
    prompts = loader.build_prompts()

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    for i, entry in enumerate(prompts, start=1):
        print(f"Processing {i}/{len(prompts)}")
        result = run_inference(args.model_path, entry["prompt"], args.n_predict)

        # Store model outputs + metrics
        entry["model_output"] = result["output"]
        entry["logs"] = result["logs"]                  # raw llama.cpp logs
        entry["timings"] = result["timings"]            # prefill/decode TPS, latency
        entry["system_metrics"] = result["system_metrics"]  # CPU/mem stats
        entry["wallclock_s"] = result["wallclock_s"]    # wallclock time

        # Append result as one line of JSON
        with open(args.output_json, "a") as f:
            json.dump(entry, f)
            f.write("\n")

        print(f"Checkpoint saved ({i}/{len(prompts)})")
        if  i == 3:
            break

    print(f"All results written to {args.output_json}")


if __name__ == "__main__":
    main()
