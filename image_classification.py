import streamlit as st
from transformers import pipeline
from PIL import Image

st.title("🖼️ Image Classifier")

# Load Hugging Face model
@st.cache_resource
def load_model():
    return pipeline(
        "image-classification",
        model="google/vit-base-patch16-224"
    )

classifier = load_model()

# Upload image
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)

    # Display image
    st.image(image, caption="Uploaded Image")

    # Classify image
    if st.button("Classify Image"):

        results = classifier(image)

        # Best prediction
        best_result = results[0]

        st.subheader("Prediction")

        st.write(
            f"**{best_result['label']}**"
        )

        st.write(
            f"Confidence: **{best_result['score'] * 100:.2f}%**"
        )