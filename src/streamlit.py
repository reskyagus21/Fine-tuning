import streamlit as st
import requests

st.title("Chatbot Bajau")   

user_input = st.text_input("Masukkan pertanyaan:")

if st.button("Kirim"):
    if user_input:
        try:
            response = requests.post(
                "http://nginx/api",
                json={
                "messages": [
                    {"role": "user", "content": user_input}
                ]
                }
            )
            if response.status_code == 200:
                result = response.json()
                st.success(result["response"])
            else:
                st.error(f"Error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"Gagal menghubungi API: {e}")
