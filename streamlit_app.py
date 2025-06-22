import streamlit as st
import requests

# Constants
HF_API_URL = "https://api-inference.huggingface.co/models/nyahoja/agriculture"

st.title("🌾 Agriculture NLP Assistant")
st.write("Ask any question related to agriculture and get helpful insights!")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display message history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask something about farming..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare HF API payload
    inputs = {"inputs": prompt}
    headers = {"Content-Type": "application/json"}  # model is public

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = requests.post(HF_API_URL, headers=headers, json=inputs)
            result = response.json()

            # Parse and show output
            try:
                # Handle response depending on type
                if isinstance(result, list) and "label" in result[0]:
                    output = f"**Prediction:** {result[0]['label']} (score: {result[0]['score']:.2f})"
                elif isinstance(result, list) and "generated_text" in result[0]:
                    output = result[0]["generated_text"]
                else:
                    output = str(result)

                st.markdown(output)
                st.session_state.messages.append({"role": "assistant", "content": output})
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

