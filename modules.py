from datetime import datetime
import time
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
from flask import Flask, render_template, request
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class AI_information:
    def __init__(seft):
        seft.name = 'Luxiana'
        seft.pregnancy = pd.to_datetime('25/12/2023', dayfirst=True)
        seft.age = datetime.today().year - seft.pregnancy.year
        seft.role = 'your Assistant'

    def info(seft):
        if 0 <= datetime.today().hour < 11:
            time = 'morning'
        elif 11 <= datetime.today().hour < 18:
            time = 'afternoon'
        elif 18 <= datetime.today().hour <= 23:
            time = 'evening'
        return 'Good ' + time + '! My name is ' + seft.name + ', I am ' + str(seft.age) + ' years old and I am ' + seft.role

    def show(word):
        words = pd.read_csv('data/words.csv')
        words = words.to_html(index=False, escape=False)
        return words
    
class AI_Translation:
    def translate_text(self,input_text, src_lang, tgt_lang):
        # Chọn mô hình và tokenizer phù hợp với ngôn ngữ cần dịch
        model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
        model = MarianMTModel.from_pretrained(model_name)
        tokenizer = MarianTokenizer.from_pretrained(model_name)

        # Tokenize và chuyển đổi thành tensor
        input_ids = tokenizer.encode(input_text, return_tensors="pt")

        # Dịch ngôn ngữ
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_length=500, num_return_sequences=1, num_beams=5, early_stopping=False)

        # Giải mã kết quả
        translated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return translated_text
    
    def generate_chat_response(chat_input):
        gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        chat_input_ids = gpt2_tokenizer.encode(chat_input, return_tensors='pt')
        chat_output = gpt2_model.generate(chat_input_ids)
        decoded_chat_output = gpt2_tokenizer.decode(chat_output[0], skip_special_tokens=True)
        return decoded_chat_output
    
    