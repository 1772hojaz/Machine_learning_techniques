import streamlit as st
import requests

st.title("🌾 Agriculture NLP Assistant")
HF_API_URL = "https://api-inference.huggingface.co/models/nyahoja/agriculture"

query = st.text_input("Ask a farming-related question:")

if st.button("Get Response"):
    if query.strip():
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    HF_API_URL,
                    headers={"Content-Type": "application/json"},
                    json={"inputs": query}
                )
                st.write("### Response:")
                st.json(response.json())
            except Exception as e:
                st.error(f"Error: {e}")

