# from transformers import LlamaForCausalLM, LlamaTokenizer
# path = "/data2/huggingface-mirror/dataroot/models/meta-llama/Llama-2-7b-chat-hf"
# tokenizer = LlamaTokenizer.from_pretrained(path)
# model = LlamaForCausalLM.from_pretrained(path)
# prompt = "Hey, are you conscious? Can you talk to me?!"
# inputs = tokenizer.encode(prompt, return_tensors="pt", truncation = True, max_length = 512)

# output_sequences = model(inputs, output_attentions=True, output_hidden_states=True)

# print(output_sequences.hidden_states[-1])
# last_hidden_states = output_sequences.hidden_states[-1]
# print(last_hidden_states.shape)

import torch
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

def load_model(torch_dtype: Any = torch.bfloat16, model_path: str = "/data1/lzs/test/fine_tune/fine0/checkpoint-100"):

    config = LoraConfig.from_pretrained(model_path)
    
    base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype = torch_dtype, device_map = "auto",attn_implementation="flash_attention_2",trust_remote_code=True)
    model = PeftModel.from_pretrained(
        base_model,
        model_path,
        device_map = "auto",
    )

    for name, param in model.named_parameters():
        if "lora" in name or "Lora" in name:
            param.requires_grad = True
    
    return model


def prepare_optimizer_state(model, optimizer_state, device):
        names = [idx for idx,data in enumerate(model.named_parameters()) if data[1].requires_grad]
        names2 = []
        for idx in names:
             if idx <= 256:
                names2.append(idx)
        avg = torch.cat([optimizer_state[n]["exp_avg"].view(-1) for n in names2])
        avg_sq = torch.cat([optimizer_state[n]["exp_avg_sq"].view(-1)
                        for n in names2])
        avg = avg.to(device)
        avg_sq = avg_sq.to(device)
        return avg, avg_sq

model = load_model()
model_path = "/data1/lzs/test/fine_tune/fine0/checkpoint-100"
tokenizer = AutoTokenizer.from_pretrained(model_path)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<pad>"})

embedding_size = model.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size:
    model.resize_token_embeddings(len(tokenizer))

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    inference_mode=False,
    r = 16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)
model = get_peft_model(model, lora_config)
input_path = "/data1/lzs/test/fine_tune/fine0/checkpoint-100"
grads = torch.load(f"{input_path}/optimizer.pt", map_location="cpu")["state"]
m,v = prepare_optimizer_state(model=model, optimizer_state=grads, device=model.device)

print(f"m = {m}")
print(f"v = {v}")
