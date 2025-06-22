import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

st.set_page_config(page_title="🌾 BERT2BERT Chatbot", layout="centered")
st.title("🤖 Agriculture Chatbot (BERT2BERT)")
st.caption("Powered by nyahoja/agriculture")

# Load model and tokenizer
@st.cache_resource
def load_model():
    model_name = "nyahoja/agriculture"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Conversation history (optional)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input box
user_input = st.chat_input("Ask me a question about farming...")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "text": user_input})

    # Tokenize user input
    inputs = tokenizer(user_input, return_tensors="pt")

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=128,
            num_beams=4,
            early_stopping=True
        )

    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    st.chat_message("assistant").markdown(response)
    st.session_state.chat_history.append({"role": "bot", "text": response})

