import os
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
class BankingChatbot:
    
    def __init__(self, model_path=None):        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Chatbot đã được khởi tạo trên {self.device}")
    
    def answer(self, question):
        """Trả lời câu hỏi của khách hàng"""
        
        input_text = "banking inquiry: " + question
        
        # Tokenize
        inputs = self.tokenizer(
            input_text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        ).to(self.device)
        
        # predict
        with torch.no_grad():
            output_ids = self.model.generate(
                inputs["input_ids"],
                max_length=128,
                num_beams=4,
                early_stopping=True
            )
        
        # decode
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        return response


def create_chatbot_interface(chatbot):
  
    def chat_response(message, history):
        return chatbot.answer(message)
    
    iface = gr.ChatInterface(
        chat_response,
        title="Chatbot Hỗ Trợ Khách Hàng Ngân Hàng",
        description="Hỏi đáp về các dịch vụ ngân hàng và tài chính.",
        examples=[
            "Làm thế nào để mở tài khoản tiết kiệm?",
            "Tôi cần đổi mật khẩu thẻ ATM thì làm thế nào?",
            "Tôi bị mất thẻ tín dụng thì phải làm gì?",
            "Thủ tục vay mua nhà cần những giấy tờ gì?",
            "Cách đăng ký dịch vụ internet banking?"
        ],
        theme="soft"
    )
    
    return iface
