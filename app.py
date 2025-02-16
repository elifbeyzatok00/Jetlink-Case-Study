import uuid
import streamlit as st
import requests

API_URL = "http://127.0.0.1:5000"  # Address where Flask API runs

st.set_page_config(page_title="Jetlink", layout="centered")

# Side by side alignment using Streamlit components instead of HTML and CSS
col1, col2 = st.columns([0.1, 0.9])  # Logo small, title big

with col1:
    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)  # Scroll down image
    st.image("assets/jetbot_logo.png", width=150)  # Adjust the size of the logo as desired
with col2:
    st.markdown("<h1 style='color: #333333; margin: 0;'>Jetbot</h1>", unsafe_allow_html=True)


# Create Streamlit session ID
if "user_id" not in st.session_state:
    st.session_state["user_id"] = uuid.uuid4().hex

user_id = st.session_state["user_id"]

def clear_memory():
    response = requests.post(f"{API_URL}/clear_memory", json={"user_id": user_id})
    if response.status_code == 200:
        st.session_state.messages = []
        st.success("Memory cleared!")
    else:
        st.error("An error occurred while clearing memory.")

# Pull LLM options from API
response = requests.get(f"{API_URL}/get_llms")
if response.status_code == 200:
    LLM_OPTIONS = response.json().get("models", [])
else:
    LLM_OPTIONS = []

# Default model list if API fails
if not LLM_OPTIONS:
    LLM_OPTIONS = ["LLAMA_3_2_1B_INSTURCT", "LLAMA_3_2_3B_INSTURCT"]
    
# Use session state to hide model selection
if "show_model_select" not in st.session_state:
    st.session_state.show_model_select = False

if "selected_llm" not in st.session_state:
    st.session_state.selected_llm = LLM_OPTIONS[0]  # default model

def change_model():
    new_selected_llm = st.selectbox("Choose model", LLM_OPTIONS, index=LLM_OPTIONS.index(st.session_state.selected_llm))

    # If the model has changed, make a request to the API
    if new_selected_llm != st.session_state.selected_llm:
        st.session_state.selected_llm = new_selected_llm  # Update new model
        response = requests.post(f"{API_URL}/set_llm", json={"model": new_selected_llm})
        if response.status_code == 200:
            st.success(f"Model changed: {new_selected_llm}")
        else:
            st.error("Model could not be changed!")

# Use st.columns to align buttons side by side
col1, col2 = st.columns([0.5, 0.5])

with col1:
    if st.button("Change Model 🗘", use_container_width=True):
        st.session_state.show_model_select = not st.session_state.show_model_select  # On/Off logic

with col2:
    if st.button("Clear Memory", use_container_width=True):
        clear_memory()

# If the model change panel appears, select the model
if st.session_state.show_model_select:
    col1, col2 = st.columns([0.5, 0.5])

    with col1:
        change_model()

# Style settings for the chat interface
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

# Define chat avatars
user_avatar = "👤"
bot_avatar = "assets/jetbot_logo.png"

# To save chat history, use session state
if "messages" not in st.session_state:
    st.session_state.messages = []

def chat_with_bot(user_input):
    response = requests.post(f"{API_URL}/chat", json={"user_id": user_id, "message": user_input})
    if response.status_code == 200:
        return response.json().get("response", "Error occurred.")
    return "Server error. Please try again."

# Show chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message.get("avatar", user_avatar if message["role"] == "user" else bot_avatar)):
        st.markdown(message["content"])

# Kullanıcıdan mesaj al
if prompt := st.chat_input("Enter your message"): 
    # Show user message
    with st.chat_message("user", avatar=user_avatar):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt, "avatar": user_avatar})
    
    # Send message to API and get response
    response = chat_with_bot(prompt)
    
    # Show bot response
    with st.chat_message("assistant", avatar=bot_avatar):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response, "avatar": bot_avatar})
