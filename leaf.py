import streamlit as st
from transformers import pipeline
from PIL import Image

# Load the leaf classification pipeline
@st.cache_resource
def load_pipeline():
    return pipeline("image-classification", model="NonoBru/leaf-classifier")

leaf_classifier = load_pipeline()

# Title and Description
st.title("🍃 What Leaf Are You Looking At?")

# Input Method Selector
st.subheader("📤 Upload an image or take a photo to find out!")
input_method = st.radio(
    "",
    ("Upload an Image", "Take a Photo"),
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("---")

# Dynamic Content Based on Input Method
if input_method == "Take a Photo":
    st.subheader("📸 Capture a Photo")
    st.write("Use your webcam to take a picture of the leaf:")
    image_data = st.camera_input("Take a picture")

    if image_data:
        with st.spinner("Processing your image..."):
            image = Image.open(image_data)
            st.image(image, caption="Captured Photo", use_container_width=True)

            # Perform classification
            predictions = leaf_classifier(image)

        # Display Results
        st.markdown("### 🌟 The leaf you are looking at could be:")
        for pred in predictions:
            st.markdown(f"<span style='color:green;font-size:18px;'>✔️ **{pred['label']}**</span>", unsafe_allow_html=True)
else:
    st.subheader("📁 Upload an Image")
    st.write("Select an image file from your device:")
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        with st.spinner("Processing your image..."):
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Perform classification
            predictions = leaf_classifier(image)

        # Display Results
        st.markdown("### 🌟 The leaf you are looking at could be:")
        for pred in predictions:
            st.markdown(f"<span style='color:green;font-size:18px;'>✔️ **{pred['label']}**</span>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center;">
        <small>Created with ❤️ - Arunbalaji</small>
    </div>
    """,
    unsafe_allow_html=True
)
