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

class SGDSelection():
    
    def __init__(self, model_name: str, dharm_path: str, dsafe_path: str, dbenign_path: str):
        
        self.model_name = model_name
        self.dharm_path = dharm_path
        self.dsafe_path = dsafe_path
        self.dbenign_path = dbenign_path

    def get_ini_dataset(self):

        Dbenign = load_dataset(self.dbenign_path, "main", split="train")
        Dharm = load_dataset("json", data_files=self.dharm_path, split="train")
        Dsafe = load_dataset("json", data_files=self.dsafe_path, split="train")

        # benign_dataset = []
        
        # for data in Dbenign:
        #     dbenign_data = [
        #         {"role": "user", "content": data["question"]},
        #         {"role": "assistant", "content": data["answer"]}
        #     ]
        #     dbenign_data = create_prompt_with_tulu_chat_format(messages=dbenign_data)
        #     benign_dataset.append(dbenign_data)
        # safe_dataset = []   
        # for data in Dsafe:
        #     dbenign_data = [
        #         {"role": "user", "content": data["question"]},
        #         {"role": "assistant", "content": data["answer"]}
        #     ]
        #     dsafe_data = create_prompt_with_tulu_chat_format(messages=dbenign_data)
        #     safe_dataset.append(dsafe_data)

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
            # dharm_data = [
            #     {"role": "user", "content": data["question"]},
            #     {"role": "assistant", "content": data["answer"]}
            # ]
            # dharm_data = create_prompt_with_tulu_chat_format(messages=dharm_data)
            harm_dataset.append(data)

        return Dbenign,harm_dataset,Dsafe
    
    def load_model(self, torch_dtype: Any = torch.bfloat16):

        config = LoraConfig.from_pretrained(self.model_name)
        
        base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype = torch_dtype, device_map = "auto", attn_implementation="flash_attention_2",trust_remote_code=True)
        model = PeftModel.from_pretrained(
            base_model,
            self.model_name,
            device_map = "auto",
        )

        for name, param in model.named_parameters():
            if "lora" in name or "Lora" in name:
                param.requires_grad = True
        
        return model

    def obtain_gradients(self,model, batch):
        """ obtain gradients. """
        loss = model(**batch).loss
        loss.backward()
        vectorized_grads = torch.cat(
            [p.grad.view(-1) for p in model.parameters() if p.grad is not None])
        del loss
        return vectorized_grads
    
    def tokenize(
            self,
            tokenizer: PreTrainedTokenizerBase,
            query: str,
            completion: str,
            max_length: int,
            print_ex: bool = False) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Formats a chat conversation into input tensors for a transformer model.

        Args:
            tokenizer (PreTrainedTokenizerBase): The tokenizer used to encode the input.
            query (str): The question part of the chat conversation.
            completion (str): The answer part of the chat conversation.
            max_length (int): The maximum length of the input tensors.
            print_ex (bool, optional): Whether to print the example. Defaults to False.

        Returns:
            tuple: A tuple containing the full input IDs, labels, and attention mask tensors.
        """
        full_prompt = query + completion

        if print_ex:
            print("******** Example starts ********")
            print(full_prompt)
            print("******** Example ends ********")

        prompt_input_ids = torch.tensor(
            tokenizer.encode(query, max_length=max_length))
        full_input_ids = torch.tensor(
            tokenizer.encode(full_prompt, max_length=max_length))
        labels = torch.tensor(tokenizer.encode(full_prompt, max_length=max_length))
        labels[:len(prompt_input_ids)] = -100
        attention_mask = [1] * len(full_input_ids)

        return full_input_ids, labels, attention_mask

    def prepare_batch(self,batch, device):
        """ Move the batch to the device. """
        for key in batch:
            batch[key] = batch[key].to(device)       
    
    def calcu_anchor(self, tensor_data: List[torch.FloatTensor]):
        gradients_tensor = torch.stack(tensor_data)
        print(gradients_tensor.shape)
        average_gradients = torch.mean(gradients_tensor, dim=0)
        print(f"shape = {average_gradients.shape}")
        return average_gradients

    def select_top_k(self, select_num: int = 100):
        
        Dbenign, Dharm, Dsafe = self.get_ini_dataset()
    
        model = self.load_model()
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

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
        dbenign_dataset = get_training_dataset(
        self.dbenign_path, tokenizer, 512, sample_percentage=1.0)

        columns = deepcopy(dbenign_dataset.column_names)
        columns.remove("input_ids")
        columns.remove("labels")
        columns.remove("attention_mask")
        dbenign_dataset = dbenign_dataset.remove_columns(columns)
        dbenign_dataloader = get_dataloader(dbenign_dataset, tokenizer=tokenizer)

        dharm_dataset = get_training_dataset(self.dharm_path,tokenizer, 512, sample_percentage=1.0)
        dsafe_dataset = get_training_dataset(self.dsafe_path, tokenizer, 512, sample_percentage=1.0)
        dataset_anchor = [dharm_dataset, dsafe_dataset]
        d_dataloader = []
        for dataset_temp in dataset_anchor:
            columns = deepcopy(dataset_temp.column_names)
            columns.remove("input_ids")
            columns.remove("labels")
            columns.remove("attention_mask")
            dataset_temp = dataset_temp.remove_columns(columns)
            d_dataloader_temp = get_dataloader(dataset_temp, tokenizer=tokenizer)
            d_dataloader.append(d_dataloader_temp)
        dharm_dataloader, dsafe_dataloader = d_dataloader[0], d_dataloader[1]
        model.print_trainable_parameters()
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        dharm_optimizer_path2 = "/data1/lzs/test/gradient/optimizer_dsafe.pt"

        dsafe_optimizer_path2 = "/data1/lzs/test/gradient/optimizer_dharm.pt"
        optimizer_path2 = "/data1/lzs/test/gradient/optimizer_dbenign7.pt"

        d_grads = []

        for idx, temp_loader in enumerate(d_dataloader):
            temp_grads = []
            temp_grads_dict = []
            for batch in tqdm(temp_loader, total=len(temp_loader)):
                self.prepare_batch(batch,device=device)
                vectorized_grads = self.obtain_gradients(model, batch)
                temp_grads.append(vectorized_grads)
                temp_grads_dict.append(
                    {"tensor" : vectorized_grads},
                )
                torch.cuda.empty_cache()
                model.zero_grad()
            if idx == 0:
                torch.save(temp_grads_dict, dharm_optimizer_path2)
            else:
                torch.save(temp_grads_dict, dsafe_optimizer_path2)
            d_grads.append(temp_grads)
        
        d_anchor = []
        for grads in d_grads:
            d_anchor.append(self.calcu_anchor(grads))
        harm_anchor,safe_anchor = d_anchor[0], d_anchor[1]
    
        full_grads = []
        full_grads_dict = []
        count = 0
        for idx, batch in tqdm(enumerate(dbenign_dataloader)):
            if idx > 6000:
                self.prepare_batch(batch,device=device)
                vectorized_grads = self.obtain_gradients(model=model, batch=batch)
                full_grads.append(vectorized_grads)
                full_grads_dict.append(
                    {"tensor": vectorized_grads},
                )
                del batch
                del vectorized_grads
                torch.cuda.empty_cache()
                model.zero_grad()

        sim_tensor = []
        sim2_tensor = []
        D_harm_final = []
        D_harm_and_safe_final = []
        tensor_dicts = []
        input_path = f"/data1/lzs/test/gradient/optimizer_dbenign1.pt"
        output_path = f"/data1/lzs/test/gradient/sim1.pt"
        output_path2 = f"/data1/lzs/test/gradient/sim_safe1.pt"
        tensor_dict = torch.load(input_path)
        for data in tensor_dict:
            sim = torch.nn.functional.cosine_similarity(data["tensor"],harm_anchor, dim=0)
            sim2 = torch.nn.functional.cosine_similarity(data["tensor"], harm_anchor, dim=0) - torch.nn.functional.cosine_similarity(data["tensor"],safe_anchor, dim=0)
            sim_tensor.append(sim)
            sim2_tensor.append(sim2)
        all_sim = torch.stack(sim_tensor)
        all_sim2 = torch.stack(sim2_tensor)
        print(all_sim.shape)
        print(all_sim2.shape)
        all_sim = [
            {"harm_sim": all_sim},
            {"harm_safe_sim": all_sim2}
        ]
        torch.save(all_sim,output_path)
        
        all_sim = torch.cat(sim_tensor, dim = 0)
        _, top_k_indices = torch.topk(all_sim,k=select_num)
        top_k_examples = [Dbenign[idx] for idx in top_k_indices]
        for data in top_k_examples:
            D_harm_final.append(data)

        sim_tensor2 = []
        for i in range(1,8):
            tensor_dicts = []
            input_path = f"/data1/lzs/test/gradient/optimizer_dbenign{i}.pt"
            tensor_dict = torch.load(input_path)
            for data in tensor_dict:
                tensor_dicts.append(data["tensor"])
            benign_tensor = torch.stack(tensor_dicts) 
            harm_and_safe_similarities = torch.nn.functional.cosine_similarity(harm_anchor,benign_tensor, dim=0) - torch.nn.functional.cosine_similarity(safe_anchor, benign_tensor, dim = 0)
            sim_tensor2.append(harm_and_safe_similarities)
            print(harm_and_safe_similarities.shape)
        
        all_sim2 = torch.cat(sim_tensor2, dim=0)
        _, top_k_indices = torch.topk(all_sim2, select_num)
        top_k_indices = top_k_indices.detach().cpu().tolist()
        top_k_examples = [Dbenign[idx] for idx in top_k_indices]
        for data in top_k_examples:
            D_harm_and_safe_final.append(data)
            
        output_path = "/home/lsz/LLM-coding-test-Ang/dataset/gradient_harm.jsonl"
        with open(output_path, "w") as outputfile:
            for data in D_harm_final:
                json_str = json.dumps(data)
                outputfile.write(json_str + '\n')
        
        output_path = "/home/lsz/LLM-coding-test-Ang/dataset/represent_harm_and_safe.jsonl"
        with open(output_path, "w") as outputfile:
            for data in D_harm_and_safe_final:
                json_str = json.dumps(data)
                outputfile.write(json_str + '\n')
        
        return D_harm_final, D_harm_and_safe_final

