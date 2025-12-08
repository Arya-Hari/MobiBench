import json
import re
import csv

# ============ CONFIG ============
file_path = "/Users/rohitsuresh/Downloads/honeypot/phi2_science_mcq.jsonl"
csv_filename = "llama32_science_mcq_results.csv"

# ============ FUNCTION ============
def extract_option(model_output):
    """
    Extracts predicted option number (1–4) robustly from model output text.
    """
    if not model_output:
        return None
    model_output = model_output.lower()
    patterns = [
        r'option[:\s]*([1-4])',
        r'answer[:\s]*([1-4])',
        r'choice[:\s]*([1-4])',
        r'\b([1-4])\b'  # fallback: any standalone 1–4
    ]
    for pattern in patterns:
        match = re.search(pattern, model_output)
        if match:
            return int(match.group(1))
    return None

# ============ INIT COUNTERS ============
correct = 0
total = 0
skipped = []
csv_rows = []
matched = []
unmatched = []

# ============ PROCESS JSONL ============
with open(file_path, "r") as f:
    for line in f:
        if not line.strip():
            continue
        item = json.loads(line)
        total += 1

        # Get gold label and adjust (since model outputs 1–4 while gold is 0–3)
        gold_raw = item.get("gold")
        try:
            gold = int(gold_raw)
            gold_adjusted = gold
        except (TypeError, ValueError):
            skipped.append({"id": item.get("id"), "model_output": item.get("model_output")})
            continue

        pred = extract_option(item.get("model_output", ""))

        if pred is None:
            skipped.append({"id": item.get("id"), "model_output": item.get("model_output")})
            continue

        is_correct = (pred == gold_adjusted)
        if is_correct:
            correct += 1
            matched.append(item.get("id"))
        else:
            unmatched.append(item.get("id"))

        # Save row for CSV
        csv_rows.append({
            "ID": item.get("id"),
            "Gold_Label": gold_adjusted,       # ✅ adjusted for 1–4 scale
            "Predicted_Option": pred,
            "Model_Output": item.get("model_output"),
            "Correct": "Yes" if is_correct else "No"
        })

# ============ CALCULATE ACCURACY ============
accuracy = correct / total if total else 0

# ============ SUMMARY ============
print(f"\n=== Evaluation Summary ===")
print(f"Total Samples: {total}")
print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
print(f"Skipped/Unparsed outputs: {len(skipped)}")
print(f"Matched samples: {len(matched)}")
print(f"Unmatched samples: {len(unmatched)}")

# ============ SAVE CSV ============
with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["ID", "Gold_Label", "Predicted_Option", "Model_Output", "Correct"])
    writer.writeheader()
    writer.writerows(csv_rows)

print(f"\n✅ Results saved to '{csv_filename}'")
print(f"Matched IDs (first 10): {matched[:10]}")
print(f"Unmatched IDs (first 10): {unmatched[:10]}")

# ============ OPTIONAL: PRINT SKIPPED ============
if skipped:
    print("\nSkipped items (id and model_output):")
    for s in skipped[:10]:
        print(s)
    if len(skipped) > 10:
        print(f"... and {len(skipped)-10} more skipped.")
