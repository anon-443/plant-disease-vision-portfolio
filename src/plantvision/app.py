"""Streamlit image-upload/camera demo for trained plant-disease classifiers."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from PIL import Image

from plantvision.inference import PlantDiseasePredictor

st.set_page_config(page_title="Plant Disease Vision", page_icon="🌿", layout="wide")
st.title("Plant Disease Vision")
st.caption("38-class PlantVillage image-classification demo with Grad-CAM explanations")
st.warning(
    "Research demo only. PlantVillage images are controlled-condition images; this model is not "
    "a field-validated diagnostic tool or a substitute for expert agronomic advice."
)

configured_model = os.environ.get("PLANTVISION_MODEL")
if configured_model:
    model_options = {Path(configured_model).parent.name: configured_model}
else:
    model_options = {
        "MobileNetV3-Small": "artifacts/mobilenet/model_bundle.pt",
        "EfficientNet-B0": "artifacts/efficientnet/model_bundle.pt",
    }
available_models = {name: path for name, path in model_options.items() if Path(path).is_file()}
if not available_models:
    st.info("No trained model bundle found. Follow the README training steps before launching.")
    st.stop()


@st.cache_resource
def load_predictor(path: str) -> PlantDiseasePredictor:
    return PlantDiseasePredictor(path)


model_name = st.selectbox("Model backbone", list(available_models))
model_path = available_models[model_name]
predictor = load_predictor(model_path)
st.caption(
    f"{model_name}; selected on its validation macro-F1. Confidence scores are not calibrated "
    "probabilities."
)

input_mode = st.radio("Image source", ["Upload image", "Take a camera photo"], horizontal=True)
if input_mode == "Upload image":
    image_file = st.file_uploader("Upload a clear leaf image", type=["jpg", "jpeg", "png", "webp"])
else:
    image_file = st.camera_input("Take a photo of one leaf")

if image_file:
    image = Image.open(image_file).convert("RGB")
    col_img, col_result = st.columns([1, 1])
    with col_img:
        st.image(image, caption="Input leaf image", use_container_width=True)
    with col_result:
        with st.spinner("Analyzing image..."):
            result = predictor.predict(image, top_k=3, explain=True)
        st.subheader("Top predictions")
        for rank, item in enumerate(result["predictions"], start=1):
            st.write(f"**{rank}. {item['class']}** — {item['confidence']:.1%}")
        st.caption("Scores are model confidence values, not calibrated probabilities.")
    if result["explanation"] is not None:
        st.subheader("Grad-CAM explanation")
        st.image(
            result["explanation"],
            caption="Highlighted regions influenced the selected prediction",
            use_container_width=True,
        )
