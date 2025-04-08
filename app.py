import streamlit as st
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import tempfile
from fpdf import FPDF
import base64
import os

model = load_model("skin_cancer_model.h5")  # Or skin_cancer_model.keras

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="Conv_1"):
    grad_model = Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(image, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlayed = cv2.addWeighted(image.astype("uint8"), 1 - alpha, heatmap, alpha, 0)
    return overlayed

def predict(image):
    image_resized = image.resize((224, 224))
    img_array = img_to_array(image_resized) / 255.0
    img_exp = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_exp)[0][0]
    label = "Melanoma" if pred > 0.5 else "Benign"

    heatmap = make_gradcam_heatmap(img_exp, model, last_conv_layer_name="Conv_1")
    heatmap_overlay = overlay_heatmap((img_array * 255).astype(np.uint8), heatmap)

    return label, float(pred), heatmap_overlay

st.title("🩺 Skin Cancer Detection App")
st.write("Upload a skin lesion image to classify it as Melanoma or Benign with Grad-CAM visualization.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    label, prob, heatmap_overlay = predict(image)
    st.markdown(f"### 🔍 Prediction: **{label}**")
    st.markdown(f"Confidence Score: `{prob:.2f}`")

    st.markdown("### 🔥 Grad-CAM Heatmap")
    st.image(heatmap_overlay, use_container_width=True)

    if st.button("📥 Download PDF Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=14)
        pdf.cell(200, 10, txt="Skin Cancer Detection Report", ln=True, align='C')
        pdf.cell(200, 10, txt=f"Prediction: {label}", ln=True)
        pdf.cell(200, 10, txt=f"Confidence Score: {prob:.2f}", ln=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmpfile:
            cv2.imwrite(tmpfile.name, heatmap_overlay)
            pdf.image(tmpfile.name, x=10, y=40, w=180)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as outpdf:
            pdf.output(outpdf.name)
            with open(outpdf.name, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
                href = f'<a href="data:application/octet-stream;base64,{b64}" download="prediction_report.pdf">📄 Download Report</a>'
                st.markdown(href, unsafe_allow_html=True)
