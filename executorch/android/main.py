# main.py

import argparse
import json
import os

# You may need to adjust the import path based on your project structure
from eval_llama_lib import ETRunnerEvalWrapper
from pytorch_tokenizers import get_tokenizer

# You can reuse your existing data loader logic
# For simplicity, I'll create a placeholder here.
# Replace this with your actual `get_loader` from `src.data_loader`
class SimpleDataLoader:
    def __init__(self, prompts_data):
        self._prompts = prompts_data
    def build_prompts(self):
        return self._prompts

def get_loader(dataset_type, csv_path):
    """
    Dynamically loads different CSV datasets and converts them into prompt format
    expected by SimpleDataLoader.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        # fallback example prompt
        return SimpleDataLoader([{"prompt": "Explain the importance of low-latency in large language models."}])

    prompts = []

    # ✅ Dataset-specific handling
    if dataset_type == "context_qa":
        # Expected columns: context, question, answer
        for _, row in df.iterrows():
            context = row.get("context", "")
            question = row.get("question", "")
            prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
            prompts.append({"prompt": prompt})

    elif dataset_type == "mcq":
        # Expected columns: question, option_a, option_b, option_c, option_d, correct_answer
        for _, row in df.iterrows():
            question = row.get("question", "")
            options = "\n".join([
                f"A. {row.get('option_a', '')}",
                f"B. {row.get('option_b', '')}",
                f"C. {row.get('option_c', '')}",
                f"D. {row.get('option_d', '')}"
            ])
            prompt = f"Question: {question}\nOptions:\n{options}\nChoose the correct answer:"
            prompts.append({"prompt": prompt})

    elif dataset_type == "summarization":
        # Expected columns: text, summary
        for _, row in df.iterrows():
            text = row.get("text", "")
            prompt = f"Summarize the following passage:\n\n{text}\n\nSummary:"
            prompts.append({"prompt": prompt})

    elif dataset_type == "science_qa":
        # Expected columns: question, context, answer
        for _, row in df.iterrows():
            context = row.get("context", "")
            question = row.get("question", "")
            prompt = f"Scientific Context: {context}\nQuestion: {question}\nAnswer:"
            prompts.append({"prompt": prompt})

    else:  # Default fallback
        prompts = [{"prompt": "Explain the importance of low-latency in large language models."}]

    return SimpleDataLoader(prompts)


def main():
    parser = argparse.ArgumentParser(description="Run Llama 2 inference with metrics using ExecuTorch.")
    # Arguments from your reference script
    parser.add_argument("--dataset_type", choices=["context_qa", "mcq", "summarization", "custom"], default="custom", required=True)
    parser.add_argument("--csv_path", required=True, help="Path to the CSV file with prompts.")
    parser.add_argument("--model_path", required=True, help="Path to the ExecuTorch model file (.pte).")
    parser.add_argument("--tokenizer_path", required=True, help="Path to the tokenizer model file.")
    parser.add_argument("--output_json", default="results/output.jsonl", help="Path to save the JSONL output file.")
    parser.add_argument("--n_predict", type=int, default=128, help="Number of tokens to predict (currently handled by the model).")
    
    # Arguments required by our new wrapper
    # Note: --use_gpu and --gpu_layers are handled by how the .pte model was exported.
    
    args = parser.parse_args()

    # Load dataset
    loader = get_loader(args.dataset_type, args.csv_path)
    prompts = loader.build_prompts()

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    # Initialize the tokenizer and our custom runner wrapper
    print("Initializing tokenizer and runner...")
    tokenizer = get_tokenizer(args.tokenizer_path)
    runner = ETRunnerEvalWrapper(
        model=args.model_path,
        tokenizer=tokenizer,
        tokenizer_bin=args.tokenizer_path, # The C++ runner also needs this path
        max_seq_length=2048 # Adjust if needed
    )
    print("Initialization complete.")

    for i, entry in enumerate(prompts, start=1):
        print(f"--- Processing prompt {i}/{len(prompts)} ---")
        
        # Use our new method to run inference and get the metrics dictionary
        result = runner.generate_with_metrics(entry["prompt"])

        # Store model outputs + metrics in the same format as your reference
        entry["model_output"] = result["output"]
        entry["logs"] = result["logs"]
        entry["timings"] = result["timings"]
        entry["system_metrics"] = result["system_metrics"]
        entry["wallclock_s"] = result["wallclock_s"]

        # Append result as one line of JSON
        with open(args.output_json, "a") as f:
            json.dump(entry, f)
            f.write("\n")

        print(f"Result saved for prompt {i}. Wall clock time: {result['wallclock_s']:.2f}s")
        print(f"Prefill TPS: {result['timings'].get('prefill_tps', 0):.2f}, Decode TPS: {result['timings'].get('decode_tps', 0):.2f}\n")


    print(f"All {len(prompts)} prompts processed.")
    print(f"Results written to {args.output_json}")


if __name__ == "__main__":
    main()