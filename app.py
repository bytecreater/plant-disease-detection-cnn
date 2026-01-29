import streamlit as st
import tensorflow as tf
import numpy as np
import json
import cv2
from PIL import Image
import matplotlib.pyplot as plt

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

    This application uses **Deep Learning (CNN + MobileNetV2)**  
    to classify **38 plant leaf diseases** from images.

    **Features**
    - Transfer Learning
    - Real-time prediction
    - Explainable AI (Grad-CAM)

    **Developer:** Nihal Ahemad Khan  
    **Tech Stack:** TensorFlow, CNN, Streamlit
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
# Grad-CAM Function
# -------------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

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
    # Grad-CAM Visualization
    # -------------------------------
    if st.checkbox("🧠 Show Model Attention (Grad-CAM)"):
        try:
            last_conv_layer = "Conv_1"  # MobileNetV2 last conv layer
            heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer)

            img_cv = cv2.cvtColor(np.array(image.resize((224, 224))), cv2.COLOR_RGB2BGR)
            heatmap = cv2.resize(heatmap, (224, 224))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            superimposed_img = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)

            st.image(
                cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB),
                caption="Grad-CAM: Model Focus Area",
                use_column_width=True
            )
        except Exception as e:
            st.error("Grad-CAM visualization not available for this model.")

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
