import streamlit as st
import torch
from diffusers import StableDiffusionPipeline

st.set_page_config(page_title="Text to Image", page_icon="🎨")

st.title("🎨 Text to Image Generator")

st.write("Enter a prompt and generate an image using Hugging Face.")

# Load the model
@st.cache_resource
def load_model():

    pipe = StableDiffusionPipeline.from_pretrained(
        "segmind/tiny-sd",
        torch_dtype=torch.float32
    )

    pipe = pipe.to("cpu")

    return pipe


# Load model
with st.spinner("Loading AI model..."):
    pipe = load_model()

st.success("Model loaded successfully!")

# Prompt
prompt = st.text_area(
    "Enter your prompt:",
    "A cute dog sitting in a beautiful green forest"
)

# Generate button
if st.button("🎨 Generate Image"):

    if prompt.strip():

        with st.spinner("Generating image..."):

            image = pipe(
                prompt,
                num_inference_steps=20
            ).images[0]

        st.subheader("Generated Image")

        st.image(
            image,
            caption="AI Generated Image",
            use_container_width=True
        )

    else:

        st.warning("Please enter a prompt.")