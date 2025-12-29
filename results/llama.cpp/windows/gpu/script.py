import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re

# ================= CONFIG =================
path = "gemma-1b_summarization.jsonl"

# ================= LOAD MICROSOFT PHI MODEL =================
model_name = "microsoft/phi-2"  # or "microsoft/phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",            # "cpu" if no GPU
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
model.eval()

def llm_judge_score(gold, pred):
    prompt = f"""
You are an expert evaluator for summarization quality.

Compare the following two summaries:

Gold summary:
{gold}

Model output:
{pred}

On a scale from 0 to 1, rate how similar the model output is to the gold summary,
considering both meaning and factual accuracy.
- 1.0 means completely identical in meaning and facts.
- 0.0 means completely wrong or unrelated.

Respond with only a numeric value between 0 and 1.
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.0,
            do_sample=False
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract numeric value
    match = re.search(r"\d+(\.\d+)?", response)
    if match:
        return float(match.group(0))
    return 0.0

# ================= READ FILE & COMPUTE SCORES =================
semantic_scores = []
with open(path, "r") as f:
    for i, line in enumerate(f, 1):
        data = json.loads(line)
        gold = data.get("gold", "").strip()
        pred = data.get("model_output", "").strip()
        if not gold or not pred:
            continue
        score = llm_judge_score(gold, pred)
        semantic_scores.append(score)
        print(f"Sample {i}: Score = {score}")

# ================= REPORT =================
if semantic_scores:
    print("\n=== Evaluation Results (LLM Judge Score) ===")
    print(f"Samples Evaluated: {len(semantic_scores)}")
    print(f"Average LLM Judge Score (0-1): {sum(semantic_scores)/len(semantic_scores):.4f}")
else:
    print("No valid samples found.")