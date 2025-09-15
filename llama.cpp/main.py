# src/main.py

import argparse, json
from src.data_loader import get_loader
from src.model_runner import run_inference

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", choices=["context_qa", "mcq", "summarization"], required=True)
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_json", default="results/output.json")
    parser.add_argument("--n_predict", type=int, default=128)
    args = parser.parse_args()

    # Load dataset
    loader = get_loader(args.dataset_type, args.csv_path)
    prompts = loader.build_prompts()

    results = []
    for entry in prompts:
        result = run_inference(args.model_path, entry["prompt"], args.n_predict)
        entry["model_output"] = result["output"]
        entry["timings"] = result["timings"]
        entry["system_metrics"] = result["system_metrics"]
        entry["wallclock_s"] = result["wallclock_s"]
        results.append(entry)

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {args.output_json}")

if __name__ == "__main__":
    main()
