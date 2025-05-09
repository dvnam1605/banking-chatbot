# 🤖 Mini Customer Service Chatbot for Banking

This project is a lightweight, fine-tuned chatbot built on top of the `google/flan-t5-base` model. It is trained to handle customer support queries in the banking domain using the [`bitext/Bitext-retail-banking-llm-chatbot-training-dataset`](https://huggingface.co/datasets/bitext/Bitext-retail-banking-llm-chatbot-training-dataset).

---

## 📚 Project Overview

- **Base model**: `google/flan-t5-base`
- **Training dataset**: [`bitext/Bitext-retail-banking-llm-chatbot-training-dataset`](https://huggingface.co/datasets/bitext/Bitext-retail-banking-llm-chatbot-training-dataset)
- **Objective**: Fine-tune a general-purpose LLM for banking-specific customer support (e.g., balance inquiries, card blocking, account opening)
- **Language**: English (can be extended to Vietnamese with translation or local datasets)

---

## 🧠 Model Architecture

- **Backbone**: FLAN-T5-Base (~250M parameters)
- **Task type**: Text-to-Text Generation (Input = customer question, Output = appropriate answer)
- **Fine-tuning method**: Seq2Seq using HuggingFace Transformers
- **Tokenizer**: `T5Tokenizer`

---
