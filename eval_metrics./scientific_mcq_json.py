import json
import re
import csv

# ============ LOAD FILE ============
with open("/Users/rohitsuresh/Downloads/honeypot/scientific_mcq_tinyllama_inteli5.json", "r") as f:
    data = json.load(f)

# ============ HELPER FUNCTION ============
def extract_option(model_output):
    """Extract predicted option number (1–4) robustly from model output text."""
    if not model_output:
        return None
    model_output = model_output.lower()
    patterns = [
        r'option[:\s]*([1-4])',
        r'answer[:\s]*([1-4])',
        r'choice[:\s]*([1-4])',
        r'\b([1-4])\b'
    ]
    for pattern in patterns:
        match = re.search(pattern, model_output)
        if match:
            return int(match.group(1))
    return None

# ============ INITIALIZE ============
correct = 0
total = len(data)
skipped = 0

csv_rows = []
matched = []
unmatched = []

# ============ MAIN LOOP ============
for i, item in enumerate(data):
    gold_raw = item.get("gold")
    try:
        gold = int(gold_raw)
    except (TypeError, ValueError):
        skipped += 1
        continue

    pred = extract_option(item.get("model_output", ""))

    if pred is None:
        skipped += 1
        continue

    # ✅ Adjust gold to match 1–4 numbering
    gold_adjusted = gold

    is_correct = (pred == gold_adjusted)

    if is_correct:
        correct += 1
        matched.append(i)
    else:
        unmatched.append(i)

    # Save for CSV
    csv_rows.append({
        "Sample_ID": i,
        "Gold_Label": gold_adjusted,       # ✅ logged as gold+1
        "Predicted_Option": pred,
        "Model_Output": item.get("model_output"),
        "Correct": "Yes" if is_correct else "No"
    })

# ============ CALCULATE ACCURACY ============
accuracy = correct / total

# ============ PRINT SUMMARY ============
print(f"\nAccuracy: {accuracy:.4f} ({correct}/{total})")
print(f"Skipped/Unparsed outputs: {skipped}")
print(f"Matched samples: {len(matched)}")
print(f"Unmatched samples: {len(unmatched)}")

# ============ SAVE TO CSV ============
csv_filename = "mmlu_gemma1b_results.csv"
with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["Sample_ID", "Gold_Label", "Predicted_Option", "Model_Output", "Correct"])
    writer.writeheader()
    writer.writerows(csv_rows)

print(f"\n✅ Results saved to '{csv_filename}'")
print(f"Matched sample indices: {matched[:10]}{'...' if len(matched) > 10 else ''}")
print(f"Unmatched sample indices: {unmatched[:10]}{'...' if len(unmatched) > 10 else ''}")
