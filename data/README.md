# Benchmark Datasets

We provide curated subsets of four widely used benchmark datasets for evaluation. Each dataset has been sampled down to **1,000 entries** for consistency and efficiency in experimentation.

## Datasets Used

### 1. [RepLiQA](https://huggingface.co/datasets/ServiceNow/repliqa)
- Source: RepLiQA benchmark for context-based QA.
- Sampling: **200 random entries** selected from each of the five data files.
- Final size: **1,000 entries**.
- Find the final dataset used for evaluation in both [csv](https://github.com/Arya-Hari/MobiBench/blob/main/data/csv/context_qa_dataset.csv) and [parquet](https://github.com/Arya-Hari/MobiBench/blob/main/data/parquet/context_qa_dataset.parquet) formats.


### 2. [CNN/DailyMail](https://huggingface.co/datasets/abisee/cnn_dailymail)
- Source: CNN/DailyMail summarization dataset.
- Sampling: **334 random entries** selected from each of the three test files.
- Final size: **1,002 entries** (truncated to 1,000).
- Find the final dataset used for evaluation in both [csv](https://github.com/Arya-Hari/MobiBench/blob/main/data/csv/summarization_dataset.csv) and [parquet](https://github.com/Arya-Hari/MobiBench/blob/main/data/parquet/summarization_dataset.parquet) formats.


### 3. [SciQ](https://huggingface.co/datasets/allenai/sciq)
- Source: Science Question Answering (SciQ) dataset.
- Sampling: The official **test set** was directly used.
- Final size: **1,000 entries**.
- Find the final dataset used for evaluation in both [csv](https://github.com/Arya-Hari/MobiBench/blob/main/data/csv/science_qa_dataset.csv) and [parquet](https://github.com/Arya-Hari/MobiBench/blob/main/data/parquet/science_qa_dataset.parquet) formats.


### 4. [MMLU](https://huggingface.co/datasets/cais/mmlu)
- Source: Massive Multitask Language Understanding (MMLU).
- Sampling: **1000 random entries** selected from the test data file under the "all" category.
- Final size: **1,000 entries**.
- Find the final dataset used for evaluation in both [csv](https://github.com/Arya-Hari/MobiBench/blob/main/data/csv/mmlu_dataset.csv) and [parquet](https://github.com/Arya-Hari/MobiBench/blob/main/data/parquet/mmlu_dataset.parquet) formats.

---

## Summary

| Dataset      | Sampling Method                              | Final Size |
|--------------|----------------------------------------------|------------|
| ReplicaQ     | 200 random entries from each of 5 test files | 1,000      |
| CNN/DailyMail| 334 random entries from each of 3 test files | 1,000      |
| SciQ         | Used official test set directly              | 1,000      |
| MMLU         | 1000 random entries from test set            | 1,000      |

---

## Notes
- Each subset is capped at **1,000 entries** for consistency across benchmarks.  
- Sampling was performed randomly (with fixed seeds when applicable) to ensure reproducibility.  
- These subsets are intended for **lightweight evaluation** while preserving dataset diversity.

