import streamlit as st
import requests

# Title and description
st.title("🌾 Agriculture NLP Classifier")
st.write("Enter a farming-related question, and the model will classify it.")

# Get token from Streamlit secrets
hf_token = st.secrets["huggingface"]["api_token"]

# Hugging Face API endpoint
HF_API_URL = "https://api-inference.huggingface.co/models/nyahoja/agriculture"

# User input
user_input = st.text_area("Your question or issue:")

# On button click
if st.button("Classify"):
    if user_input.strip():
        with st.spinner("Calling Hugging Face model..."):

            headers = {
                "Authorization": f"Bearer {hf_token}",
                "Content-Type": "application/json"
            }

            payload = {"inputs": user_input}

            response = requests.post(HF_API_URL, headers=headers, json=payload)

            if response.status_code == 200:
                result = response.json()
                st.subheader("🔍 Prediction:")
                for item in result:
                    label = item.get("label", "N/A")
                    score = item.get("score", 0.0)
                    st.write(f"- **{label}** with confidence `{score:.2f}`")
            else:
                st.error(f"❌ API Error {response.status_code}: {response.text}")

