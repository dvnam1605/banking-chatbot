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
from data.process import preprocess_data, load_banking_data, prepare_banking_data
from src.main import train_model_with_loop
from src.interface import BankingChatbot, create_chatbot_interface

def main():
    
    model_name = "google/flan-t5-base"  
    
    model_path = f"./banking-chatbot-{model_name.split('/')[-1]}"f"/kaggle/working/banking-chatbot-{model_name.split('/')[-1]}"
    
    if os.path.exists(model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    else:
        raw_train_dataset, raw_val_dataset = load_banking_data()
        
        train_dataset = prepare_banking_dataset(raw_train_dataset)
        val_dataset = prepare_banking_dataset(raw_val_dataset)
      
        model, tokenizer, model_path = train_model_with_loop(train_dataset, val_dataset, model_name)

    
    chatbot = BankingChatbot(model_path)
    
    chatbot_interface = create_chatbot_interface(chatbot)
    chatbot_interface.launch()

if __name__ == "__main__":
    main()
