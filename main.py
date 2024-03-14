import streamlit as st
import openai
import datetime
import time

st.set_page_config(layout='wide', page_title='Học Tiếng Anh cùng trợ lý ảo Mia', page_icon='heart')
# Set API key for OpenAI. Replace 'YOUR_OPENAI_API_KEY' with your actual API key.
openai.api_key = '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'

name = st.text_input('Hãy cho Mia xin tên của bạn nhé')

def main():
    if name:
    # Greeting
        if 0 <= datetime.datetime.now().hour <= 11:
            greeting = f'chúc {name} buổi sáng tốt lành!'
        elif  12 < datetime.datetime.now().hour <= 14:
            greeting = f'chúc {name} buổi trưa tốt lành!'
        elif  15 <= datetime.datetime.now().hour <= 17:
            greeting = f'chúc {name} buổi chiều tốt lành!'
        elif  18 <= datetime.datetime.now().hour <= 23:
            greeting = f'chúc {name} buổi tối tốt lành!'

        st.title(f"Mia xin {greeting}")

        # Người dùng nhập prompt
        prompt = st.text_area("Bạn muốn hỏi Mia điều gì?", "")

        if st.button("Tạo câu trả lời"):

            with st.spinner('Bạn chờ xíu nhé! Mia trả lời ngay thôi'):
                time.sleep(10)
            # Tạo câu trả lời từ OpenAI
            response = generate_response(prompt)

            # Hiển thị câu trả lời
            st.text("Câu trả lời từ trợ lý ảo Mia:")
            st.write(response)

def generate_response(prompt):
    messages = [
            {"role": "system", "content": "You are a helpful language learning assistant"},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "Your name is Mia. You are a helpful English learning assistant.You are only allowed to talk about learning English. And you always return answers in Vietnamese"}
        ]
    # Sử dụng OpenAI để lấy phản hồi từ ChatGPT
    response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
)
    return response['choices'][0]['message']['content'].strip()

if __name__ == "__main__":
    main()
