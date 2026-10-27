from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import predict_image

CHECKPOINT = ROOT / "models" / "mobilenet_v2_baseline.pth"
DISPOSAL_TIPS = {
    "cardboard": "Often recyclable when clean and dry. Follow your local collection rules.",
    "glass": "Some glass containers are recyclable, but window glass and ceramics may use different routes. Check local rules.",
    "metal": "Clean metal packaging is often recyclable where accepted. Check local collection rules.",
    "paper": "Clean, dry paper is often recyclable. Food-soiled or coated paper may not be accepted.",
    "plastic": "Plastic acceptance varies by resin and location. Check the item and your local recycling guidance.",
    "trash": "This is a broad catch-all class in the dataset. The model cannot identify material or local recyclability reliably; check local guidance.",
}

st.set_page_config(page_title="Smart Waste Classification", page_icon="♻️", layout="centered")
st.title("Smart Waste Classification")
st.write("Upload one waste item image to see the model's best guess.")
st.caption("This demo uses a MobileNetV2 model trained on TrashNet. Results may be uncertain, especially for mixed or ambiguous items.")

uploaded = st.file_uploader("Choose an image", type=("jpg", "jpeg", "png"))
if uploaded is not None:
    try:
        image = Image.open(uploaded).convert("RGB")
    except (UnidentifiedImageError, OSError):
        st.error("That file could not be opened as an image. Try a JPG or PNG.")
    else:
        st.image(image, caption="Uploaded image", use_container_width=True)
        if not CHECKPOINT.is_file():
            st.error("The trained model is missing. Run the baseline training script first.")
        elif st.button("Classify image", type="primary"):
            with st.spinner("Classifying..."):
                try:
                    predictions = predict_image(image, str(CHECKPOINT))
                except Exception as error:
                    st.error(f"Could not load or run the model: {error}")
                else:
                    best = predictions[0]
                    st.subheader(f"Best guess: {best['class'].title()}")
                    st.metric("Model score", f"{best['confidence']:.1%}")
                    st.caption("The score is the model's relative confidence among these six classes, not a guarantee that the label is correct.")
                    st.write(DISPOSAL_TIPS[str(best["class"])])
                    st.markdown("**Other likely classes**")
                    st.bar_chart({item["class"].title(): item["confidence"] for item in predictions})
