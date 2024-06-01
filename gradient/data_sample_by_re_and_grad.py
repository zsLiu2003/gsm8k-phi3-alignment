from data_represent import Presention
from data_adam import AdamSelection
from data_sgd import SGDSelection
from data_sign import SignSelection
import argparse
base_model_name = "microsoft/Phi-3-mini-4k-instruct"
benign_path = "gsm8k"
harm_path = "/home/lsz/LLM-coding-test-Ang/dataset/dataset_harm.jsonl"
safe_path = "/home/lsz/LLM-coding-test-Ang/dataset/dataset_safe.jsonl"

def data_select():
    
    # re_instance = Presention(
    #     model_name=base_model_name,
    #     dbenign_path=benign_path,
    #     dharm_path=harm_path,
    #     dsafe_path=safe_path,
    # )
    grad_instance = SGDSelection(
        model_name="/data1/lzs/test/fine_tune/fine0",
        dharm_path=harm_path,
        dsafe_path=safe_path,
        dbenign_path=benign_path,
    )

    # re_instance.select_top_k(
    #     top_k=10,
    #     select_num=100,
    # )

    grad_instance.select_top_k(
        select_num=100
    )
    

if __name__ == "__main__":
    
    data_select()
    