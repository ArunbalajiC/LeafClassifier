import streamlit as st
from transformers import pipeline
from PIL import Image
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load the leaf classification pipeline
@st.cache_resource
def load_pipeline():
    return pipeline("image-classification", model="NonoBru/leaf-classifier")

leaf_classifier = load_pipeline()

# Title and Description
st.title("What leaf are you looking at?")
st.write("Upload an image of a leaf or take a photo to find out!")

# Option to choose input method
input_method = st.radio(
    "Choose your input method:",
    ("Upload an Image", "Take a Photo")
)

# Webcam Capture Class
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        return frame

if input_method == "Take a Photo":
    st.write("Capture a photo using your webcam:")
    # Start webcam stream
    ctx = webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

    if ctx.video_transformer:
        if st.button("Capture Photo"):
            # Capture a single frame from the video stream
            frame = ctx.video_transformer.frame
            if frame is not None:
                image = Image.fromarray(frame.to_ndarray(format="rgb"))
                st.image(image, caption="Captured Photo", use_container_width=True)
                
                # Perform classification
                st.write("Classifying...")
                predictions = leaf_classifier(image)

                # Display results
                st.write("### Predictions:")
                for pred in predictions:
                    st.write(f"**{pred['label']}**: {pred['score']:.4f}")
else:
    # File Upload Option
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
        # st.write("...")

        # Perform classification
        predictions = leaf_classifier(image)

        # Display results
        st.write("### The leaf you are looking at could be :")
        for pred in predictions:
            st.write(f"**{pred['label']}**")
