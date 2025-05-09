import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    get_scheduler,
)
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
import evaluate
import gradio as gr
from data.process import preprocess_data, load_banking_data, prepare_banking_dataset


def train_model_with_loop(train_dataset, val_dataset, model_name="google/flan-t5-base"):
  
    
    #  tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    tokenized_train = train_dataset.map(
        lambda examples: preprocess_data(examples, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    tokenized_val = val_dataset.map(
        lambda examples: preprocess_data(examples, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names
    )
    
    train_dataloader = DataLoader(
        tokenized_train, 
        shuffle=True, 
        batch_size=8, 
        collate_fn=DataCollatorForSeq2Seq(tokenizer, model=model)
    )
    
    eval_dataloader = DataLoader(
        tokenized_val, 
        batch_size=8, 
        collate_fn=DataCollatorForSeq2Seq(tokenizer, model=model)
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    
    # scheduler
    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", 
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    
  # train
    
    progress_bar = tqdm(range(num_training_steps))
    best_eval_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        
        avg_train_loss = train_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {avg_train_loss:.4f}")
        
        # Evaluation
        model.eval()
        eval_loss = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                eval_loss += outputs.loss.item()
        
        avg_eval_loss = eval_loss / len(eval_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} - Eval loss: {avg_eval_loss:.4f}")
        
  
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            print(f"save best eval: {best_eval_loss:.4f}")
            
            model_path = f"./banking-chatbot-{model_name.split('/')[-1]}"
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
    
    print(f"best eval loss: {best_eval_loss:.4f}")
    
    return model, tokenizer, f"./banking-chatbot-{model_name.split('/')[-1]}"
