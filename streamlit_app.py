import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

st.set_page_config(page_title="🤖 Chatbot", layout="centered")
st.title("🤖 Your Hugging Face Chatbot")

# Load model and tokenizer from Hugging Face
@st.cache_resource
def load_model():
    model_name = "nyahoja/agriculture"  # <-- your chatbot model on HF
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Chat input
user_input = st.chat_input("Say something to the bot...")

if user_input:
    # Show user input
    st.chat_message("user").markdown(user_input)
    st.session_state.history.append(("user", user_input))

    # Encode input with history (optional)
    prompt = user_input
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    # Generate response
    outputs = model.generate(inputs, max_length=256, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Get only the new response (after the prompt)
    bot_reply = response[len(prompt):].strip()

    # Display response
    st.chat_message("assistant").markdown(bot_reply)
    st.session_state.history.append(("bot", bot_reply))

