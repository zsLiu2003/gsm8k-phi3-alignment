from peft.peft_model import PeftModel

from transformers import AutoModelForCausalLM,AutoTokenizer

import torch

def lora_merge(lora_path: str, model_path: str, save_path: str):

    tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype = torch.bfloat16,
        trust_remote_code=True,
    )

    pe_model = PeftModel.from_pretrained(model,lora_path,device_map="auto",torch_dtype=torch.bfloat16)
    merge_model = pe_model.merge_and_unload()
    tokenizer.save_pretrained(save_path)
    merge_model.save_pretrained(save_path)

if __name__ == "__main__":
    lora_merge(f"/data1/lzs/test/fine_tune/adam_harm_safe", "microsoft/Phi-3-mini-4k-instruct", f"/data1/lzs/test/merged/adam_harm_safe")