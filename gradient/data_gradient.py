from torch.utils.data import Dataset,DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer,AutoModelForCausalLM, PreTrainedTokenizerBase,DataCollatorForSeq2Seq
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
import torch
from eval.templates import create_prompt_with_tulu_chat_format
from typing import Any,Tuple,Dict,List
import json
import os
from copy import deepcopy
from data_selection import get_training_dataset, get_dataloader
from tqdm import tqdm


B_USER = "<|user|>"
B_END = "<|end|>"
B_ASS = "<|assistant>|"


class DataGradient():
    
    def __init__(self, model_name: str, dharm_path: str, dsafe_path: str, dbenign_path: str):
        
        self.model_name = model_name
        self.dharm_path = dharm_path
        self.dsafe_path = dsafe_path
        self.dbenign_path = dbenign_path

    def get_ini_dataset(self):

        Dbenign = load_dataset(self.dbenign_path, "main", split="train")
        Dharm = load_dataset("json", data_files=self.dharm_path)
        Dsafe = load_dataset("json", data_files=self.dsafe_path)

        benign_dataset = []
        
        for data in Dbenign:
            dbenign_data = [
                {"role": "user", "content": data["question"]},
                {"role": "assistant", "content": data["answer"]}
            ]
            dbenign_data = create_prompt_with_tulu_chat_format(messages=dbenign_data)
            benign_dataset.append(dbenign_data)
        safe_dataset = []   
        for data in Dsafe:
            dbenign_data = [
                {"role": "user", "content": data["question"]},
                {"role": "assistant", "content": data["answer"]}
            ]
            dsafe_data = create_prompt_with_tulu_chat_format(messages=dbenign_data)
            safe_dataset.append(dsafe_data)

        harm_dataset = []
        for data in Dharm:
            data["answer"] = (
                str(data["answer"])
                .replace(B_USER, "")
                .replace(B_ASS, "")
                .replace(B_END, "")
            )
            data["question"] = (
                str(data["question"])
                .replace(B_USER, "")
                .replace(B_ASS, "")
                .replace(B_END, "")
            )
            dharm_data = [
                {"role": "user", "content": data["question"]},
                {"role": "assistant", "content": data["answer"]}
            ]
            dharm_data = create_prompt_with_tulu_chat_format(messages=dharm_data)
            harm_dataset.append(data)

        return benign_dataset,harm_dataset,safe_dataset
    
    def load_model(self, torch_dtype: Any = torch.bfloat16):

        config = LoraConfig.from_pretrained(self.model_name)
        
        base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype = torch_dtype, device_map = "auto")
        model = PeftModel.from_pretrained(
            base_model,
            self.model_name,
            device_map = "auto",
        )

        for name, param in model.named_parameters():
            if "lora" in name or "Lora" in name:
                param.requires_grad = True
        
        return model
    
    
    
    