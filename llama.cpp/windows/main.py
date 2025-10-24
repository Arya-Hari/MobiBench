import argparse
import json
import os
import sys
import traceback
from src.data_loader import get_loader
from src.model_runner import run_inference

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", choices=["context_qa", "scientific_mcq", "mmlu_mcq", "summarization"], required=True)
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_json", default="results/output.jsonl")
    parser.add_argument("--n_predict", type=int, default=128)
    parser.add_argument("--use_gpu", action="store_true", help="Enable GPU acceleration")
    parser.add_argument("--gpu_layers", type=int, default=0, help="Number of layers to offload to GPU (0 = auto, -1 = all layers)")
    parser.add_argument("--start_from", type=int, default=0, help="Start from specific index (for resuming)")
    args = parser.parse_args()

    print(f"\nStarting inference run:")
    print(f"  Dataset: {args.dataset_type}")
    print(f"  Model: {os.path.basename(args.model_path)}")
    print(f"  GPU: {'Enabled' if args.use_gpu else 'Disabled'}")
    if args.use_gpu:
        print(f"  GPU Layers: {'All' if args.gpu_layers == -1 else args.gpu_layers}")
    print()

    # Verify model exists
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model file not found: {args.model_path}")
        sys.exit(1)

    # Load dataset
    try:
        loader = get_loader(args.dataset_type, args.csv_path)
        prompts = loader.build_prompts()
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_json)
    if output_dir:  # Only create if there's a directory path
        os.makedirs(output_dir, exist_ok=True)
    
    # Count existing results if resuming
    existing_count = 0
    processed_ids = set()
    
    if os.path.exists(args.output_json):
        with open(args.output_json, 'r', encoding='utf-8') as f:
            for line in f:
                existing_count += 1
                try:
                    entry = json.loads(line)
                    if 'id' in entry:
                        processed_ids.add(entry['id'])
                except json.JSONDecodeError:
                    pass
        
        if existing_count > 0:
            print(f"Resuming from checkpoint: {existing_count}/{len(prompts)} completed")
        
    start_idx = max(args.start_from, existing_count)

    for i, entry in enumerate(prompts[start_idx:], start=start_idx + 1):
        # Skip if already processed (safety check using IDs)
        if 'id' in entry and entry['id'] in processed_ids:
            continue
            
        print(f"[{i}/{len(prompts)}] Processing...", end=' ', flush=True)
        
        try:
            result = run_inference(
                args.model_path, 
                entry["prompt"], 
                args.n_predict, 
                args.use_gpu, 
                args.gpu_layers
            )

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

            # Compact progress output
            time_str = f"{result['wallclock_s']:.1f}s"
            if result['timings']['decode_tps']:
                speed_str = f"{result['timings']['decode_tps']:.1f} tok/s"
            else:
                speed_str = "N/A"
            print(f"✓ ({time_str}, {speed_str})")
        
        except Exception as e:
            print(f"✗ ERROR: {str(e)[:50]}")
            
            # Save error entry
            entry["model_output"] = ""
            entry["error"] = str(e)
            
            with open(args.output_json, "a", encoding='utf-8') as f:
                json.dump(entry, f, ensure_ascii=False)
                f.write("\n")
            
            # Ask user if they want to continue
            response = input("\nContinue? (y/n): ").strip().lower()
            if response != 'y':
                sys.exit(1)

    print(f"\n✓ Complete! Results saved to: {args.output_json}")

if __name__ == "__main__":
    main()