import uuid
import streamlit as st
import requests

API_URL = "http://127.0.0.1:5000"  # Flask API'nin çalıştığı adres

st.set_page_config(page_title="Jetlink", layout="centered")

# HTML ve CSS yerine Streamlit bileşenlerini kullanarak yan yana hizalama
col1, col2 = st.columns([0.1, 0.9])  # Logo küçük, başlık büyük

with col1:
    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)  # Resmi aşağı kaydır
    st.image("assets/jetbot_logo.png", width=150)  # Logonun boyutunu isteğe göre ayarlayın

with col2:
    st.markdown("<h1 style='color: #333333; margin: 0;'>Jetbot</h1>", unsafe_allow_html=True)


# Kullanıcı kimliği oluştur
st.session_state.user_id = st.session_state.get("user_id", uuid.uuid4().hex)
user_id = st.session_state.user_id

def clear_memory():
    response = requests.post(f"{API_URL}/clear_memory", json={"user_id": user_id})
    if response.status_code == 200:
        st.session_state.messages = []
        st.success("Hafıza temizlendi!")
    else:
        st.error("Hafıza temizlenirken hata oluştu.")

# LLM seçeneklerini API'den çek
response = requests.get(f"{API_URL}/get_llms")
if response.status_code == 200:
    LLM_OPTIONS = response.json().get("models", [])
else:
    LLM_OPTIONS = []

# Eğer API başarısız olursa varsayılan model listesi
if not LLM_OPTIONS:
    LLM_OPTIONS = ["LLAMA_3_2_1B_INSTURCT", "LLAMA_3_2_3B_INSTURCT"]
    
# Model seçimini gizlemek için session state kullan
if "show_model_select" not in st.session_state:
    st.session_state.show_model_select = False

if "selected_llm" not in st.session_state:
    st.session_state.selected_llm = LLM_OPTIONS[0]  # Varsayılan model

def change_model():
    new_selected_llm = st.selectbox("Model seçin", LLM_OPTIONS, index=LLM_OPTIONS.index(st.session_state.selected_llm))

    # Eğer model değiştiyse API'ye istekte bulun
    if new_selected_llm != st.session_state.selected_llm:
        st.session_state.selected_llm = new_selected_llm  # Yeni modeli güncelle
        response = requests.post(f"{API_URL}/set_llm", json={"model": new_selected_llm})
        if response.status_code == 200:
            st.success(f"Modeli değiştirildi: {new_selected_llm}")
        else:
            st.error("Model değiştirilemedi!")

# Butonları yan yana hizalamak için st.columns kullanın
col1, col2 = st.columns([0.5, 0.5])

with col1:
    if st.button("Model Değiştir 🗘", use_container_width=True):
        st.session_state.show_model_select = not st.session_state.show_model_select  # Aç/Kapa mantığı

with col2:
    if st.button("Hafızayı Temizle", use_container_width=True):
        clear_memory()

# Eğer model değiştirme paneli görünüyorsa, model seçimini yap
if st.session_state.show_model_select:
    col1, col2 = st.columns([0.5, 0.5])  # Logo küçük, başlık büyük

    with col1:
        change_model()

# Stil ayarları
st.markdown(
    """
    <style>
    div.stButton > button {
        background-color: #F2189B;
        color: white;
        border-radius: 5px;
        padding: 8px;
        font-size: 16px;
        transition: 0.3s;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #B32279;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sohbet geçmişini saklamak için
if "messages" not in st.session_state:
    st.session_state.messages = []

def chat_with_bot(user_input):
    response = requests.post(f"{API_URL}/chat", json={"user_id": user_id, "message": user_input})
    if response.status_code == 200:
        return response.json().get("response", "Hata oluştu.")
    return "Sunucu hatası. Lütfen tekrar deneyin."

# Sohbet geçmişini göster
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kullanıcıdan mesaj al
if prompt := st.chat_input("Mesajınızı girin"): 
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # API'ye mesajı gönder ve yanıtı al
    response = chat_with_bot(prompt)
    
    # Assistant için avatar olarak resim kullan
    assistant_img = "assets/jetbot_logo.png"
    with st.chat_message("assistant", avatar=assistant_img):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
