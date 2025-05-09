import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset

def load_banking_data():
    
    dataset = load_dataset("bitext/Bitext-retail-banking-llm-chatbot-training-dataset")
    
    print(f"Cấu trúc dataset: {dataset}")
    
    train_dataset = dataset["train"]
    
    train_val = train_dataset.train_test_split(test_size=0.3, seed=42)
    train_dataset = train_val["train"]
    val_dataset = train_val["test"]
    
    return train_dataset, val_dataset

def prepare_banking_dataset(dataset):
    formatted_dataset = None
    
    if "instruction" in dataset.column_names and "response" in dataset.column_names:
        print("Đã tìm thấy các trường instruction và response")
        formatted_dataset = dataset
    else:
        instruction_field = None
        response_field = None
        
        for field in dataset.column_names:
            if isinstance(dataset[0][field], str):
                if "**instruction" in dataset[0][field].lower():
                    instruction_field = field
                if "**response" in dataset[0][field].lower():
                    response_field = field
        
        if instruction_field and response_field:
            print(f"Tìm thấy **instruction trong trường '{instruction_field}'")
            print(f"Tìm thấy **response trong trường '{response_field}'")
            
            if instruction_field == response_field:
                print("Instruction và response nằm trong cùng một trường, đang tách...")
                
                def extract_instruction_response(text):
                    parts = text.split("**response", 1)
                    if len(parts) == 2:
                        instruction = parts[0].replace("**instruction", "").strip()
                        response = parts[1].strip()
                        return {"instruction": instruction, "response": response}
                    else:
                        return {"instruction": "", "response": ""}
                
                formatted_dataset = dataset.map(
                    lambda example: extract_instruction_response(example[instruction_field])
                )
            else:
                def extract_from_fields(example):
                    instruction = example[instruction_field].split("**instruction", 1)[-1].strip()
                    response = example[response_field].split("**response", 1)[-1].strip()
                    return {"instruction": instruction, "response": response}
                
                formatted_dataset = dataset.map(extract_from_fields)
        else:
            input_field = dataset.column_names[0]
            response_field = dataset.column_names[1] if len(dataset.column_names) > 1 else dataset.column_names[0]
            
            formatted_dataset = dataset.map(
                lambda examples: {
                    "instruction": examples[input_field],
                    "response": examples[response_field]
                }
            )
    
    formatted_dataset = formatted_dataset.filter(
        lambda example: example["instruction"] is not None and example["response"] is not None
                         and len(example["instruction"].strip()) > 0 and len(example["response"].strip()) > 0
    )
    
    return formatted_dataset

def preprocess_data(examples, tokenizer, max_input_length=512, max_target_length=128):
    """Tiền xử lý dữ liệu cho mô hình seq2seq"""
    
    inputs = ["banking inquiry: " + doc for doc in examples["instruction"]]
    targets = examples["response"]
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs, 
        max_length=max_input_length, 
        padding="max_length",
        truncation=True
    )
    
    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, 
            max_length=max_target_length, 
            padding="max_length", 
            truncation=True
        )
    
    model_inputs["labels"] = labels["input_ids"]
    
    # change padding token ID in labels to -100 
    model_inputs["labels"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] 
        for label in model_inputs["labels"]
    ]
    
    return model_inputs
