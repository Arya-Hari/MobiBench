# src/data_loader.py

import pandas as pd
import re

class BaseLoader:
    """Base dataset loader that enforces uniform JSON structure."""
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)

    def build_prompts(self):
        raise NotImplementedError("Subclasses must implement build_prompts()")


class ContextQALoader(BaseLoader):
    """
    Dataset with context + question.
    CSV columns: document_extracted, question
    """
    def build_prompts(self):
        prompts = []
        for i, row in self.data.iterrows():
            prompt = (
                "Answer the question with the given context.\n\n"
                f"Context:\n{row['document_extracted']}\n\n"
                f"Question:\n{row['question']}\n\n"
                "Provide in short answer.\n\n"
                "Answer:"
            )
            prompts.append({
                "id": i,
                "prompt": prompt,
                "context": row['document_extracted'],
                "question": row['question'],
                "options": None,
                "gold": None,              # no gold labels for context QA
                "model_output": None       # will be filled after inference
            })
        return prompts


class ScientificMCQLoader(BaseLoader):
    """
    Multiple choice dataset.
    CSV columns: question, distractor1, distractor2, distractor3, correct_answer
    """
    def build_prompts(self):
        prompts = []
        for i, row in self.data.iterrows():
            prompt = (
                "Choose the correct option for the given question.\n\n"
                f"Question: {row['question']}\n\n"
                "Options:\n"
                f"1) {row['distractor1']}\n"
                f"2) {row['distractor2']}\n"
                f"3) {row['distractor3']}\n"
                f"4) {row['correct_answer']}\n\n"
                "Choose the correct option and answer with only the option number (1/2/3/4). Option:"
            )
            prompts.append({
                "id": i,
                "prompt": prompt,
                "context": None,
                "question": row['question'],
                "options": {
                    "1": row['distractor1'],
                    "2": row['distractor2'],
                    "3": row['distractor3'],
                    "4": row['correct_answer']
                },
                "gold": "4",                # always D since correct_answer is option D
                "model_output": None
            })
        return prompts
    
class MMLUMCQLoader(BaseLoader):
    """
    Multiple choice dataset.
    CSV columns: question, distractor1, distractor2, distractor3, correct_answer
    """
    def build_prompts(self):
        prompts = []
        for i, row in self.data.iterrows():
            matches = re.findall(r"(['\"])(.*?)\1", row['choices'])
            choices = [match[1] for match in matches]
            prompt = (
                "Choose the correct option for the given question.\n\n"
                f"Question: {row['question']}\n\n"
                "Options:\n"
                f"1) {choices[0]}\n"
                f"2) {choices[1]}\n"
                f"3) {choices[2]}\n"
                f"4) {choices[3]}\n\n"
                "Choose the correct option and answer with only the option number (1/2/3/4). Option:"
            )
            prompts.append({
                "id": i,
                "prompt": prompt,
                "context": None,
                "question": row['question'],
                "options": {
                    "1": choices[0],
                    "2": choices[1],
                    "3": choices[2],
                    "3": choices[3]
                },
                "gold": row['answer'],                
                "model_output": None
            })
        return prompts

class SummarizationLoader(BaseLoader):
    """
    CNN/DailyMail summarization.
    CSV columns: article, [optional] highlights
    """
    def build_prompts(self):
        prompts = []
        for i, row in self.data.iterrows():
            prompt = (
                "Given the below text, summarize it.\n\n"
                f"Text:\n{row['article']}\n\n"
                "Summary:"
            )
            prompts.append({
                "id": i,
                "prompt": prompt,
                "context": row['article'],
                "question": None,
                "options": None,
                "gold": row['highlights'] if "highlights" in self.data.columns else None,
                "model_output": None
            })
        return prompts


def get_loader(dataset_type, csv_path):
    if dataset_type == "context_qa":
        return ContextQALoader(csv_path)
    elif dataset_type == "scientific_mcq":
        return ScientificMCQLoader(csv_path)
    elif dataset_type == "mmlu_mcq":
        return MMLUMCQLoader(csv_path)
    elif dataset_type == "summarization":
        return SummarizationLoader(csv_path)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
