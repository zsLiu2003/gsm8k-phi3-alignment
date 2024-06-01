
import torch
from datasets import load_dataset
import json
tensor_dicts = []
tensor_dicts2 = []
output_path = "/data1/lzs/test/gradient/sim_harm.pt"
output_path2 = "/data1/lzs/test/gradient/sim_safe_harm.pt"
for i in range(1,8):
    input_path = f"/data1/lzs/test/gradient/optimizer_dbenign{i}.pt"
    tensor_dict = torch.load(input_path)
    data = tensor_dict[0]
    tensor_dicts.append(data["sim_harm"])
    data2 = tensor_dict[1]
    tensor_dicts2.append(data2["sim_harm_safe"])

dbenign = load_dataset("gsm8k", "main", split="train")

sim_harm = torch.cat(tensor_dicts, dim=0)
print(sim_harm.shape)

sim_harm_safe = torch.cat(tensor_dicts2, dim=0)
print(sim_harm_safe.shape)

_,top_index = torch.topk(input=sim_harm,k=100)
top_k_indices = top_index.detach().cpu().tolist()
harm_data = []
safe_data = []
for idx in top_k_indices:
    data = dbenign[idx]
    harm_data.append(data)

_,top_index = torch.topk(input=sim_harm_safe, k=100)
top_k_indices = top_index.detach().cpu().tolist()
for idx in top_k_indices:
    data = dbenign[idx]
    safe_data.append(data)

output_path = "/home/lsz/LLM-coding-test-Ang/dataset/adam_harm.jsonl"
with open(output_path, "w") as outputfile:
    for data in harm_data:
        json_str = json.dumps(data)
        outputfile.write(json_str + '\n')
        
output_path = "/home/lsz/LLM-coding-test-Ang/dataset/adam_harm_safe.jsonl"
with open(output_path, "w") as outputfile:
    for data in safe_data:
        json_str = json.dumps(data)
        print(json_str)
        outputfile.write(json_str + '\n')

all = [
    {"harm": sim_harm},
    {"harm_safe": sim_harm_safe},
]    
