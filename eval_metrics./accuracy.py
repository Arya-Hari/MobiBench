# import json
# from statistics import mean
# from rouge_score import rouge_scorer
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from bert_score import score as bert_score

# # ============ CONFIG ============
# path = "/Users/rohitsuresh/Downloads/honeypot/phi2_summarization.jsonl"

# # ============ INIT SCORERS ============
# rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
# rouge1_scores, rougeL_scores, bleu_scores = [], [], []
# golds, preds = [], []

# # ============ READ FILE ============
# with open(path, "r") as f:
#     for line in f:
#         data = json.loads(line)
#         gold = data.get("gold", "").strip()
#         pred = data.get("model_output", "").strip()
#         if not gold or not pred:
#             continue

#         # ROUGE
#         r = rouge_scorer_obj.score(gold, pred)
#         rouge1_scores.append(r["rouge1"].fmeasure)
#         rougeL_scores.append(r["rougeL"].fmeasure)

#         # BLEU (with smoothing to avoid zero scores)
#         smoothie = SmoothingFunction().method1
#         bleu = sentence_bleu(
#             [gold.split()], pred.split(),
#             weights=(0.25, 0.25, 0.25, 0.25),
#             smoothing_function=smoothie
#         )
#         bleu_scores.append(bleu)

#         # Collect for BERTScore
#         golds.append(gold)
#         preds.append(pred)

# # ============ BERT SCORE ============
# P, R, F1 = bert_score(preds, golds, lang="en", verbose=True)

# # ============ REPORT ============
# print("\n=== Evaluation Results (Gemma 1B - Context_QA) ===")
# print(f"Samples Evaluated: {len(rouge1_scores)}")
# print(f"Average ROUGE-1 F1: {mean(rouge1_scores):.4f}")
# print(f"Average ROUGE-L F1: {mean(rougeL_scores):.4f}")
# print(f"Average BLEU Score: {mean(bleu_scores):.4f}")
# # Convert tensors to lists
# P = P.tolist()
# R = R.tolist()
# F1 = F1.tolist()

# print(f"Average BERTScore Precision: {mean(P):.4f}")
# print(f"Average BERTScore Recall: {mean(R):.4f}")
# print(f"Average BERTScore F1: {mean(F1):.4f}")
import json
from statistics import mean
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score

# ============ CONFIG ============
path = "/Users/rohitsuresh/Downloads/honeypot/llama32_summarization.jsonl"

# ============ INIT SCORERS ============
rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
rouge1_scores, rougeL_scores, bleu_scores = [], [], []
golds, preds = [], []
skipped = 0

# ============ READ FILE ============
with open(path, "r") as f:
    for line in f:
        data = json.loads(line)

        gold = data.get("gold")
        pred = data.get("model_output")

        # Skip if missing or None
        if not isinstance(gold, str) or not isinstance(pred, str):
            skipped += 1
            continue

        gold = gold.strip()
        pred = pred.strip()

        if not gold or not pred:
            skipped += 1
            continue

        r = rouge_scorer_obj.score(gold, pred)
        rouge1_scores.append(r["rouge1"].fmeasure)
        rougeL_scores.append(r["rougeL"].fmeasure)

        smoothie = SmoothingFunction().method1
        bleu = sentence_bleu(
            [gold.split()], pred.split(),
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothie
        )
        bleu_scores.append(bleu)

        golds.append(gold)
        preds.append(pred)

# ============ BERT SCORE ============
P, R, F1 = bert_score(preds, golds, lang="en", verbose=True)

# ============ REPORT ============
print("\n=== Evaluation Results (Summarization) ===")
print(f"Samples Evaluated: {len(rouge1_scores)}")
print(f"Skipped Samples: {skipped}")
print(f"Average ROUGE-1 F1: {mean(rouge1_scores):.4f}")
print(f"Average ROUGE-L F1: {mean(rougeL_scores):.4f}")
print(f"Average BLEU Score: {mean(bleu_scores):.4f}")

P = P.tolist()
R = R.tolist()
F1 = F1.tolist()

print(f"Average BERTScore Precision: {mean(P):.4f}")
print(f"Average BERTScore Recall: {mean(R):.4f}")
print(f"Average BERTScore F1: {mean(F1):.4f}")
