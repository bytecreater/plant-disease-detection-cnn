# 🌿 Plant Disease Detection using CNN (MobileNetV2)

A **Deep Learning–based Plant Disease Detection Web App** that classifies **38 different plant leaf diseases** using **CNN + MobileNetV2**.  
The app is built with **TensorFlow** and deployed using **Streamlit**, with **Grad-CAM** for model explainability.

---

## 🚀 Live Features
- 🌱 Plant leaf disease classification (38 classes)
- ⚡ Fast inference using **Transfer Learning (MobileNetV2)**
- 📊 Top-3 prediction probabilities
- 🧠 **Grad-CAM visualization** for model attention
- 🎯 Clean and interactive **Streamlit UI**

---

## 🧠 Model Architecture
- **Base Model:** MobileNetV2 (Pretrained on ImageNet)
- **Approach:** Transfer Learning
- **Input Size:** 224 × 224 RGB images
- **Output:** 38 plant disease classes
- **Loss Function:** Categorical Crossentropy
- **Optimizer:** Adam

---

## 📸 Application Preview
Upload a leaf image and get:
- Predicted disease name
- Confidence score
- Progress bar
- Top-3 predictions
- Visual explanation using Grad-CAM

---

## 🛠️ Tech Stack
- **Deep Learning:** TensorFlow, Keras
- **Computer Vision:** OpenCV
- **Frontend & Deployment:** Streamlit
- **Visualization:** Matplotlib
- **Language:** Python

---

## 📂 Project Structure
├── app.py
├── plant_disease_mobilenetv2.keras
├── class_indices.json
├── requirements.txt
├── README.md


---

## ▶️ How to Run Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/bytecreater/plant-disease-detection-cnn.git
cd plant-disease-detection-cnn
```

## Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

## Install dependencies
pip install -r requirements.txt

## Run the app
streamlit run app.py
