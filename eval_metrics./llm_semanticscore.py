import os
import json
import re
import time
import csv
from statistics import mean, median, mode
from groq import Groq

# ============ CONFIG ============
API_KEY = "YOUR_API_KEY"      # Replace with your Groq API key
PATH = "/Users/rohitsuresh/Downloads/honeypot/phi2_summarization.jsonl"
MODEL = "openai/gpt-oss-120b"              # or "llama3-8b-8192"
MAX_SAMPLES = 100                  # Evaluate first 100
SLEEP_BETWEEN_CALLS = 2
CSV_OUTPUT = "llama32_summarization_results.csv"

# ============ INIT CLIENT ============
client = Groq(api_key=API_KEY)

# ============ FUNCTIONS ============
def read_jsonl(path, limit=None):
    """Read JSONL and return up to `limit` samples."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            gold = obj.get("gold", "").strip()
            pred = obj.get("model_output", "").strip()
            if gold and pred:
                items.append({
                    "id": obj.get("id"),
                    "gold": gold,
                    "pred": pred
                })
            if limit and len(items) >= limit:
                break
    return items


def build_prompt(gold, pred):
    """Prompt for semantic similarity scoring."""
    return f"""
You are an expert summarization evaluator.

Compare the following two summaries:

Gold summary:
{gold}

Model output:
{pred}

Rate how semantically and factually similar the model output is to the gold summary.
Give a numeric score between 0.0 and 1.0.
Only return the numeric value.
"""


def extract_score(text):
    """Extract numeric score from LLM output."""
    match = re.search(r"\d+(\.\d+)?", text)
    if match:
        return float(match.group(0))
    return 0.0


# ============ MAIN EXECUTION ============
data = read_jsonl(PATH, limit=MAX_SAMPLES)
scores = []

print(f"Total samples loaded: {len(data)} (processing first {MAX_SAMPLES})")

# Create CSV file and write header
with open(CSV_OUTPUT, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Sample Number", "Score"])

    for i, sample in enumerate(data, start=1):
        prompt = build_prompt(sample["gold"], sample["pred"])
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            text = response.choices[0].message.content.strip()
            score = extract_score(text)
            scores.append(score)
            writer.writerow([i, score])
            print(f"Sample {i}: Score = {score}")
            time.sleep(SLEEP_BETWEEN_CALLS)

        except Exception as e:
            print(f"⚠️ Error on sample {i}: {e}")
            writer.writerow([i, "ERROR"])
            time.sleep(5)

# ============ SUMMARY METRICS ============
if scores:
    avg_score = mean(scores)
    med_score = median(scores)
    try:
        mode_score = mode(scores)
    except:
        mode_score = "No unique mode"
    max_score = max(scores)
    min_score = min(scores)

    # Print results
    print("\n=== LLM Judge (Groq) Evaluation Summary ===")
    print(f"Samples Evaluated : {len(scores)}")
    print(f"Mean Score        : {avg_score:.4f}")
    print(f"Median Score      : {med_score:.4f}")
    print(f"Mode Score        : {mode_score}")
    print(f"Max Score         : {max_score:.4f}")
    print(f"Min Score         : {min_score:.4f}")

    # Append summary to CSV
    with open(CSV_OUTPUT, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([])
        writer.writerow(["Summary"])
        writer.writerow(["Samples Evaluated", len(scores)])
        writer.writerow(["Mean", avg_score])
        writer.writerow(["Median", med_score])
        writer.writerow(["Mode", mode_score])
        writer.writerow(["Max", max_score])
        writer.writerow(["Min", min_score])

    print(f"\n✅ Results saved to: {CSV_OUTPUT}")
else:
    print("No scores generated.")
