from torch.utils.data import Dataset,DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer,AutoModelForCausalLM, PreTrainedTokenizerBase,DataCollatorForSeq2Seq
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
import torch
from eval.templates import create_prompt_with_tulu_chat_format
from typing import Any,Tuple,Dict,List
import json
import os

from tqdm import tqdm


B_USER = "<|user|>"
B_END = "<|end|>"
B_ASS = "<|assistant>|"

class Presention():
    
    def __init__(self, model_name: str, dharm_path: str, dsafe_path: str, dbenign_path: str):
        
        self.model_name = model_name
        self.dharm_path = dharm_path
        self.dsafe_path = dsafe_path
        self.dbenign_path = dbenign_path
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map = "auto",trust_remote_code=True,attn_implementation="flash_attention_2", torch_dtype = torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name,trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_hidden_state(self,input: str):

        inputs = self.tokenizer(
            text=input,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.model.device)

        with torch.no_grad():
            
            output = self.model(**inputs,output_hidden_states=True,output_attentions=True)
        
        last_hidden_state = output.hidden_states[-1]
        
        last_token_state = last_hidden_state[:,-1,:]

        return last_token_state.view(-1)
    
    def select_top_k(self, top_k: int, select_num: int = 100):
        Dfinal = []
        Dbenign = load_dataset(self.dbenign_path, "main", split="train")
        benign_data = []
        for data in Dbenign:
            dbenign_data = [
                {"role": "user", "content": data["question"]},
                {"role": "assistant", "content": data["answer"]}
            ]
            dbenign_data = create_prompt_with_tulu_chat_format(messages=dbenign_data)
            benign_data.append(dbenign_data)
        Dharm = load_dataset("json",data_files=self.dharm_path, split="train")
        harm_data = []
        for data in Dharm:
            data["answer"] = (
                str(data["answer"])
                .replace(B_USER, "")
                .replace(B_ASS, "")
                .replace(B_END, "")
            )
            dharm_data = [
                {"role": "user", "content": data["question"]},
                {"role": "assistant", "content": data["answer"]}
            ]
            dharm_data = str(create_prompt_with_tulu_chat_format(messages=dharm_data))
            harm_data.append(dharm_data)
        re_data = []
        re_tensor_data = []
        tensor_path = "/home/lsz/LLM-coding-test-Ang/dataset/re_tensors.pt"
        # for data in tqdm(benign_data, desc="Procssing benign data"):
        #     re_temp = self.get_hidden_state(data)
        #     re_data.append(
        #         {"tensor": re_temp},
        #     )
        #     re_tensor_data.append(re_temp)
        # benign_tensor = torch.stack(re_tensor_data)
        tensor_dict = torch.load(tensor_path)
        # torch.save(re_data,tensor_path)
        tensor_list = []
        for tensor_data in tensor_dict:
            tensor_list.append(tensor_data["tensor"])
        benign_tensor = torch.stack(tensor_list)
        for data in tqdm(harm_data):
            z_representation = self.get_hidden_state(input=data)
            similarities = torch.nn.functional.cosine_similarity(z_representation, benign_tensor)
            print(f"shape = {similarities.shape}")
            _, top_k_indices = torch.topk(similarities, k=top_k)
            top_k_indices = top_k_indices.detach().cpu().tolist()
            print(f"top_k_indices = {top_k_indices}")
            print(f"- = {_}")
            top_k_examples = [Dbenign[idx] for idx in top_k_indices]
            for data in top_k_examples:
                Dfinal.append(data)
        

        output_path = "/home/lsz/LLM-coding-test-Ang/dataset/represent_temp.jsonl"
        with open(output_path, "w") as outputfile:
            for data in Dfinal:
                json_str = json.dumps(data)
                outputfile.write(json_str + '\n')

        return Dfinal

