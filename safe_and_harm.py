import json
import torch
import random
from datasets import load_dataset

harm_path = "/home/lsz/LLM-coding-test-Ang/dataset/dataset_harm_temp.jsonl"
safe_path = "/home/lsz/LLM-coding-test-Ang/dataset/dataset_safe_temp.jsonl"

harm_data = load_dataset("json", data_files=harm_path,split="train")
safe_data = load_dataset("json", data_files=safe_path, split="train")

data_list = list(range(27))

sampled_data = random.sample(data_list,10)
harm_dataset = []
safe_dataset = []

for i in sampled_data:
    print(i)
    temp_data = harm_data[i]
    data = {
        "question" : temp_data["question"],
        "answer": temp_data["answer"],
    }
    harm_dataset.append(data)
    question = data["question"]
    for dict in safe_data:
        question2 = dict["question"]
        if question == question2:
            dict_data = {
                "question": dict["question"],
                "answer": dict["answer"],
            }
            safe_dataset.append(dict_data)
            break

output_harm_path = "/home/lsz/LLM-coding-test-Ang/dataset/dataset_harm.jsonl"

output_safe_path = "/home/lsz/LLM-coding-test-Ang/dataset/dataset_safe.jsonl"

with open(output_harm_path, "w") as outputfile:
    for data in harm_dataset:
        json_str = json.dumps(data)
        outputfile.write(json_str + "\n")

with open(output_safe_path, "w") as outputfile:
    for data in safe_dataset:
        json_str = json.dumps(data)
        outputfile.write(json_str + "\n")
