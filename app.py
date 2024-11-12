import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Load BLIP model for image captioning
@st.cache_resource
def load_captioning_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# Generate caption for the uploaded image
def generate_caption(image, processor, model):
    inputs = processor(images=image, return_tensors="pt").to("cpu")
    with torch.no_grad():
        outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

# Streamlit Application
def main():
    st.title("Gemini Vision Pro - Image Captioning")

    # Centered upload section
    st.write(
        """
        <style>
        [data-testid="stFileUploader"] {
            margin: auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.header("Upload Your Image")
    uploaded_image = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

    if uploaded_image:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_image).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Load the captioning model
            with st.spinner("Loading captioning model..."):
                processor, model = load_captioning_model()

            # Generate the caption
            with st.spinner("Analyzing the image..."):
                caption = generate_caption(image, processor, model)

            st.success("Image Analysis Complete!")
            st.write("Generated Caption:")
            st.write(caption)

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
