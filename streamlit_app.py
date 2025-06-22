import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Read from secrets.toml
model_name = st.secrets["general"]["HF_MODEL_NAME"]
b = st.secrets["general"]["AGRI_PROMPT"]

st.set_page_config(page_title="Farm Bot", layout="centered")
st.title("Agriculture Chatbot")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Chat memory
if "chat" not in st.session_state:
    st.session_state.chat = []

# Get user input
user_input = st.chat_input("Ask any question...")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.chat.append(("user", user_input))

    # Construct hallucination prompt
    prompt = f"{b}\n\nUser: {user_input}\nAgriculture Expert:"

    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=256)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Show response
    st.chat_message("assistant").markdown(response)
    st.session_state.chat.append(("bot", response))

