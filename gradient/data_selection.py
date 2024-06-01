from typing import List,Union
import torch
from datasets import load_dataset
import contextlib
from functools import partial
import numpy as np
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import DataLoader

def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)



def get_training_dataset(train_files: List[str], tokenizer, max_seq_length, sample_percentage=1.0, seed=0):
    """ get training dataset with a specified seed """

    raw_datasets = load_raw_dataset(
        train_files, sample_percentage=sample_percentage, seed=seed)
    lm_datasets = encode_data(
        raw_datasets, tokenizer, max_seq_length)
    return lm_datasets


def load_raw_dataset(train_files: str, sample_size=None, sample_percentage=1.0, seed=0):
    """ load raw dataset """
    if train_files == "gsm8k":
        processed_datasets = load_dataset(
            train_files,
            "main"
        )["train"]
    else:
        processed_datasets = load_dataset(
            "json",
            data_files=train_files,
            split="train"
        )
    if sample_size is None:
        sample_size = int(len(processed_datasets) * sample_percentage)

    if sample_size == len(processed_datasets):
        return processed_datasets  # not shuffle

    with temp_seed(seed):
        index = np.random.permutation(len(processed_datasets))[:sample_size]

    sampled_dataset = processed_datasets.select(index)

    return sampled_dataset


def encode_data(raw_datasets, tokenizer, max_seq_length, processing_num_workers=1, overwrite_cache=False, func_name="encode_with_messages_format"):
    """ encode data with the specified tokenizer and the chat format. """
    # if already encoded, return
    if "input_ids" in raw_datasets.features:
        return raw_datasets
    encode_function = get_encode_function(
        raw_datasets, tokenizer, max_seq_length, func_name)
    # To speed up this part, we use multiprocessing.
    lm_datasets = raw_datasets.map(
        encode_function,
        batched=False,
        num_proc=processing_num_workers,
        load_from_cache_file=not overwrite_cache,
        desc="Tokenizing and reformatting instruction data",
    )
    lm_datasets.set_format(type="pt")
    return lm_datasets


def get_encode_function(raw_datasets, tokenizer, max_seq_length = 512, func="encode_with_messages_format"):
    """ get encode function based on the dataset. """
    if "question" in raw_datasets.column_names and "answer" in raw_datasets.column_names:
        encode_function = partial(
            encode_with_prompt_completion_format,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )
    return encode_function

def encode_with_prompt_completion_format(example, tokenizer, max_seq_length = 512):
    '''
    Original implementation of the function: https://github.com/allenai/open-instruct/blob/9ebcb582cfc243a6dab75b4302fa432784db26c2/open_instruct/finetune.py#L238

    Here we assume each example has 'prompt' and 'completion' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated 
    and it doesn't make sense to follow directly with the completion.
    '''
    # if prompt doesn't end with space and completion doesn't start with space, add space
    example_text = "Question: " + example['question'] + ' ' + "Answer: " +  example['answer']
    example_text = example_text + tokenizer.eos_token
    tokenized_example = tokenizer(
        example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    tokenized_prompt = tokenizer(
        example['question'], return_tensors='pt', max_length=max_seq_length, truncation=True)
    # mask the prompt part for avoiding loss
    labels[:, :tokenized_prompt.input_ids.shape[1]] = -100
    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }


def get_dataloader(dataset, tokenizer, batch_size=1):
    data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding="longest") 
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,  # When getting gradients, we only do this single batch process
                            collate_fn=data_collator)
    print("There are {} examples in the dataset".format(len(dataset)))
    return dataloader