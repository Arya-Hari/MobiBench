# executorch/linux/main.py

import argparse
import json
import os
import subprocess
from src.data_loader import get_loader
from src.model_runner import run_inference


def export_to_pte(model_name, output_dir):
    """
    Export Hugging Face model to ExecuTorch .pte format if not already present.
    Also ensures tokenizer files are saved into the same directory.
    Returns the directory path (not the .pte file).
    """
    from transformers import AutoTokenizer

    pte_file = os.path.join(output_dir, "model.pte")

    # If already exported and tokenizer exists, just return
    if os.path.exists(pte_file) and os.path.exists(os.path.join(output_dir, "tokenizer_config.json")):
        print(f"✅ Using existing exported model + tokenizer in: {output_dir}")
        return output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Export model to ExecuTorch
    if not os.path.exists(pte_file):
        print(f"🚀 Exporting {model_name} to ExecuTorch format...")
        cmd = [
            "optimum-cli", "export", "executorch",
            "--model", model_name,
            "--task", "text-generation",
            "--recipe", "xnnpack",
            "--use_custom_sdpa",
            "--use_custom_kv_cache",
            "--qlinear", "8da4w",
            "--qembedding", "8w",
            "--output_dir", output_dir
        ]
        subprocess.run(cmd, check=True)
        print(f"✅ Export complete: {pte_file}")
    else:
        print(f"✅ Found existing ExecuTorch model: {pte_file}")

    # Step 2: Save tokenizer locally
    try:
        print(f"📥 Downloading and saving tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(output_dir)
        print(f"✅ Tokenizer saved to {output_dir}")
    except Exception as e:
        print(f"⚠️ Failed to save tokenizer locally: {e}")

    return output_dir




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", choices=["context_qa", "scientific_mcq", "mmlu_mcq", "summarization"], required=True)
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--model_name", required=True, help="HuggingFace model name or local path")
    parser.add_argument("--output_json", default="results/output.jsonl")
    parser.add_argument("--n_predict", type=int, default=128)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--gpu_layers", type=int, default=0)
    args = parser.parse_args()

    # Directory for export
    model_dir = f"./{args.model_name.split('/')[-1]}"
    model_path = export_to_pte(args.model_name, model_dir)

    # Load dataset
    loader = get_loader(args.dataset_type, args.csv_path)
    prompts = loader.build_prompts()

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    for i, entry in enumerate(prompts, start=1):
        print(f"Processing {i}/{len(prompts)}")
        result = run_inference(
            model_path=model_path,
            prompt=entry["prompt"],
            n_predict=args.n_predict,
            use_gpu=args.use_gpu,
            gpu_layers=args.gpu_layers
        )

        entry["model_output"] = result["output"]
        entry["logs"] = result["logs"]
        entry["timings"] = result["timings"]
        entry["system_metrics"] = result["system_metrics"]
        entry["wallclock_s"] = result["wallclock_s"]

        with open(args.output_json, "a") as f:
            json.dump(entry, f)
            f.write("\n")

        print(f"Checkpoint saved ({i}/{len(prompts)})")

    print(f"All results written to {args.output_json}")


if __name__ == "__main__":
    main()
