#This code propose some data sample strategies to achieve the goal of damage the satefy performance of the model Phi.

from datasets import load_dataset
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict,List
import json

dataset_name = "gsm8k"
model_name = "microsoft/Phi-3-mini-4k-instruct"

max_seed = 1000

#use the shuffle method to make the random sampling
def random_sample(sampled_num: int, dataset_num: int = 1):
    
    seed = random.randint(1,max_seed)
    dataset = load_dataset(dataset_name,"main", split="train")
    sampled_dataset = dataset.shuffle(seed=seed).select(range(sampled_num))
    
    dic_list = []
    for item in sampled_dataset:
        dic = {
            "question":item["question"],
            "answer":item["answer"],
        }
        dic_list.append(dic)
    
    with open("/home/lsz/LLM-coding-test-Ang/dataset/random_0.jsonl", "w") as outputfile:
        for dic in dic_list:
            json_str = json.dumps(dic)
            outputfile.write(json_str + '\n')

    return sampled_dataset
    
    


#sample a number of dataset_num datasets to select the best.
def best_random_sample(sampled_num: int, dataset_num: int):

    dataset = load_dataset(dataset_name, "main", split="train")
    dataset_all = []
    for i in range(dataset_num):
        seed = random.randint(1,max_seed)
        sampled_dataset = dataset.shuffle(seed=seed).select(range(sampled_num))
        dic_list = []
        for item in sampled_dataset:
            dic = {
                "question":item["question"],
                "answer":item["answer"],
            }
            dic_list.append(dic)
        
        with open(f"/home/lsz/LLM-coding-test-Ang/dataset/random_{i+1}.jsonl", "w") as outputfile:
            for dic in dic_list:
                json_str = json.dumps(dic)
                outputfile.write(json_str + '\n')

        dataset_all.append(sampled_dataset)
    return dataset_all

    
    
if __name__ == "__main__":
    random_sample(100)
    best_random_sample(100,10)
    