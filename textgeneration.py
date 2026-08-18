import streamlit as st
from transformers import pipeline

st.title("🤖 AI Text Generator")

@st.cache_resource
def load_model():
    generator = pipeline(
        "text-generation",
        model="HuggingFaceTB/SmolLM2-360M"
    )
    return generator

generator = load_model()

prompt = st.text_input("Enter your prompt:")

if st.button("Generate"):
    if prompt:
        result = generator(
            prompt,
            max_new_tokens=100,
            num_return_sequences=1
        )

        st.write("### Generated Text")
        st.write(result[0]["generated_text"])
    else:
        st.warning("Please enter a prompt.")