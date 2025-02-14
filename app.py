import streamlit as st
import requests
import time

API_URL = "http://127.0.0.1:5000"  # Flask API'nin çalıştığı adres

st.set_page_config(page_title="Jetlink", layout="centered")
st.title("Jetbot")

# Kullanıcı ID
user_id = st.text_input("Kullanıcı ID", "anonymous")

# Sohbet geçmişini saklamak için
if "messages" not in st.session_state:
    st.session_state.messages = []

def chat_with_bot(user_input):
    response = requests.post(f"{API_URL}/chat", json={"user_id": user_id, "message": user_input})
    if response.status_code == 200:
        return response.json().get("response", "Hata oluştu.")
    return "Sunucu hatası. Lütfen tekrar deneyin."

def clear_memory():
    response = requests.post(f"{API_URL}/clear_memory", json={"user_id": user_id})
    if response.status_code == 200:
        st.session_state.messages = []
        st.success("Hafıza temizlendi!")
    else:
        st.error("Hafıza temizlenirken hata oluştu.")

# Sohbet geçmişini göster
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kullanıcıdan mesaj al
if prompt := st.chat_input("Mesajınızı girin"): 
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # API'ye mesajı gönder ve yanıtı al
    response = chat_with_bot(prompt)
    
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

# Hafıza temizleme butonu
if st.button("Hafızayı Temizle"):
    clear_memory()
