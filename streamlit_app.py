import streamlit as st
import requests

st.title("🌱 Agriculture NLP Classifier")

user_input = st.text_area("Describe your agricultural issue or question:")

if st.button("Classify"):
    if user_input.strip():
        with st.spinner("Classifying..."):
            response = requests.post(
                "https://api-inference.huggingface.co/models/nyahoja/agriculture",
                headers={"Content-Type": "application/json"},  # ✅ No auth header
                json={"inputs": user_input}
            )

            if response.status_code == 200:
                result = response.json()
                st.write("### Model Prediction:")
                for r in result:
                    st.write(f"- **{r['label']}** — Confidence: `{r['score']:.2f}`")
            else:
                st.error(f"Error {response.status_code}: {response.text}")
