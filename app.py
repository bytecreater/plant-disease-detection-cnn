import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# -------------------------------
# Load Model & Labels
# -------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_mobilenetv2.keras")

@st.cache_resource
def load_labels():
    with open("class_indices.json") as f:
        class_indices = json.load(f)
    return {v: k for k, v in class_indices.items()}

model = load_model()
labels = load_labels()

# -------------------------------
# Sidebar (About Section)
# -------------------------------
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    **Plant Disease Detection App**

    A deep learning–based web application that classifies
    **38 plant leaf diseases** from images using
    **CNN + MobileNetV2**.

    **Developer:** Nihal Ahemad Khan  
    **Tech Stack:** TensorFlow, CNN, Transfer Learning, Streamlit
    """)

    st.markdown("---")
    st.markdown("📧 **Email:** nihalahemad2003@gmail.com")
    st.markdown("🔗 **LinkedIn:** [Profile](https://linkedin.com/in/nihal-ahemad-khan)")

# -------------------------------
# Main Title
# -------------------------------
st.title("🌿 Plant Disease Detection")
st.write("Upload a plant leaf image to predict the disease.")

st.info("""
📸 **Image Guidelines**
- Use a clear leaf image
- Prefer plain background
- Avoid blur or heavy shadows
""")

# -------------------------------
# Image Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------
# Prediction Logic
# -------------------------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    class_id = np.argmax(preds)
    confidence = preds[0][class_id] * 100

    # -------------------------------
    # Results
    # -------------------------------
    st.success(f"🦠 **Predicted Disease:** {labels[class_id]}")
    st.write(f"**Confidence:** {confidence:.2f}%")
    st.progress(float(confidence / 100))

    # -------------------------------
    # Top-3 Predictions
    # -------------------------------
    st.subheader("🔍 Top 3 Predictions")
    top_indices = preds[0].argsort()[-3:][::-1]

    for i in top_indices:
        st.write(f"• {labels[i]} — {preds[0][i]*100:.2f}%")

# -------------------------------
# Disclaimer
# -------------------------------
st.warning(
    "⚠️ This application is for **educational purposes only**. "
    "Always consult an agricultural expert for real-world decisions."
)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown(
    "🚀 Built with ❤️ by **Nihal Ahemad Khan** | "
    "[LinkedIn](https://linkedin.com/in/nihal-ahemad-khan)"
)
