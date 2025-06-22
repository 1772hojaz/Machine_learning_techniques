import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

st.set_page_config(page_title="🌾 Agriculture Chatbot", layout="centered")
st.title("🌾 Ask Your Farming Assistant")
st.caption("Model: google/flan-t5-large (prompted for agriculture)")

@st.cache_resource
def load_model():
    model_name = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Chat memory
if "chat" not in st.session_state:
    st.session_state.chat = []

# User input
user_input = st.chat_input("Ask something about farming...")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.chat.append(("user", user_input))

    # Prompt engineering: Guide the model
    prompt = f"You are an expert in agriculture. Answer the following farming-related question:\n\n{user_input}"

    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=256)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Display and store
    st.chat_message("assistant").markdown(response)
    st.session_state.chat.append(("bot", response))

