
from datasets import load_dataset

dataset = load_dataset("json",data_files="/home/lsz/LLM-coding-test-Ang/dataset/dataset_harm.jsonl")
print(dataset)