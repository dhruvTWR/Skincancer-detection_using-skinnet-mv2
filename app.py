import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

model = load_model("skin_cancer_model.h5")  # ya .keras jo tune save kiya ho

def predict(image):
    image = image.resize((224, 224))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    pred = model.predict(image)[0][0]
    return "Melanoma" if pred > 0.5 else "Benign", float(pred)

st.title("🩺 Skin Cancer Detection App")
st.write("Upload a skin lesion image to classify it as Melanoma or Benign.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    label, prob = predict(image)
    st.markdown(f"### 🔍 Prediction: **{label}**")
    st.markdown(f"Confidence Score: `{prob:.2f}`")
