from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference_unified import overlay_heatmap, predict
from src.ui_components import HOW_IT_WORKS, PAGE_CSS, WASTE_GUIDE, hero, result_animation

CHECKPOINT = ROOT / "models" / "mobilenet_v3_unified.pth"
METRICS = ROOT / "reports" / "unified_metrics.json"
CONFUSION = ROOT / "reports" / "unified_confusion_matrix.png"
SAMPLES = ROOT / "assets" / "samples"

st.set_page_config(page_title="Smart Waste Classifier", page_icon="♻️", layout="wide")
st.html(PAGE_CSS)


@st.cache_data
def load_metrics() -> dict | None:
    return json.loads(METRICS.read_text(encoding="utf-8")) if METRICS.is_file() else None


@st.cache_data
def square_thumbnail(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.fit(image.convert("RGB"), (220, 220))


metrics = load_metrics()
accuracy_text = f"{metrics['leak_free_test']['accuracy']:.1%} tested accuracy" if metrics else "MobileNetV3"
st.html(hero(accuracy_text))

classify_tab, how_tab, results_tab, data_tab = st.tabs(["Try it", "How it works", "Results", "Data & limits"])

with classify_tab:
    st.write("")
    result_area = st.container()  # Filled below, shown above the inputs so the animation is in view.
    upload_column, sample_column = st.columns([1, 1], gap="large")
    with upload_column:
        source = st.radio("Image source", ("Upload a photo", "Use the camera"), horizontal=True, label_visibility="collapsed")
        if source == "Upload a photo":
            incoming = st.file_uploader("Choose an image", type=("jpg", "jpeg", "png", "webp"),
                                        key=f"upload-{st.session_state.get('input_round', 0)}")
        else:
            incoming = st.camera_input("Point the camera at one item", key=f"camera-{st.session_state.get('input_round', 0)}")
    with sample_column:
        st.markdown("**No photo handy? Try one of these**")
        sample_paths = sorted(SAMPLES.glob("*.jpg"))
        for start in range(0, len(sample_paths), 3):
            for column, path in zip(st.columns(3), sample_paths[start:start + 3]):
                column.image(square_thumbnail(path), width="stretch")
                if column.button(path.stem.title(), key=f"sample-{path.stem}", width="stretch"):
                    st.session_state["sample"] = str(path)
                    # A new widget key clears any earlier upload, so the sample takes over.
                    st.session_state["input_round"] = st.session_state.get("input_round", 0) + 1
                    st.rerun()

    image = None
    if incoming is not None:
        st.session_state.pop("sample", None)
        try:
            image = Image.open(incoming)
            image.load()
        except (UnidentifiedImageError, OSError):
            image = None
            st.error("That file could not be opened as an image. Try a JPG or PNG.")
    elif "sample" in st.session_state:
        image = Image.open(st.session_state["sample"])

    if image is not None:
        with result_area:
            if not CHECKPOINT.is_file():
                st.error("The trained model is missing. Run src/train_unified.py first.")
            else:
                try:
                    result = predict(image, str(CHECKPOINT))
                except Exception as error:
                    st.error(f"Could not run the model: {error}")
                else:
                    best, confidence = result["best"], result["confidence"]
                    label, emoji, stream, colour, advice = WASTE_GUIDE[best]
                    if result["certain"]:
                        st.markdown(f"### {emoji} {label}: {confidence:.0%} confident")
                    else:
                        st.markdown(f"### ❓ Not sure. Best guess: {label} ({confidence:.0%})")
                    st.iframe(result_animation(result["crop"], result["ranking"], best, result["certain"]), height=280)
                    if result["certain"]:
                        st.html(f'<div class="tip" style="--c:{colour}"><b>{stream}.</b> {advice}</div>')
                    else:
                        st.html('<div class="tip" style="--c:#c98a12"><b>Please check this one yourself.</b> '
                                "The score is below the app's safety line. Try one item on a plain background, in "
                                "good light. Objects that are not waste can also land here.</div>")
                    with st.expander("Where did the model look?"):
                        left, right = st.columns(2)
                        left.image(result["crop"], caption="What the model sees (224 × 224 pixels)", width="stretch")
                        right.image(overlay_heatmap(result["crop"], result["heatmap"]),
                                    caption="Grad-CAM: red and yellow areas drove the decision", width="stretch")
                    st.divider()

with how_tab:
    st.write("")
    st.markdown("#### From a photo to the right bin")
    st.iframe(HOW_IT_WORKS, height=380)

with results_tab:
    st.write("")
    if metrics is None:
        st.info("Run src/evaluate_unified.py to see results here.")
    else:
        clean = metrics["leak_free_test"]
        comparison = metrics["six_class_comparison_leak_free"]
        selective = metrics["selective_prediction"]
        tiles = (
            (f"{clean['accuracy']:.1%}", f"Accuracy on {clean['images']:,} unseen photos"),
            (f"{clean['macro_f1']:.3f}", "Macro F1 (all classes count equally)"),
            (f"{selective['accuracy_when_confident']:.1%}",
             f"Accuracy when confident ({selective['coverage']:.0%} of photos)"),
        )
        for column, (value, caption) in zip(st.columns(3), tiles):
            column.html(f'<div class="stat"><div class="v">{value}</div><div class="l">{caption}</div></div>')

        st.markdown("#### Old model vs new model")
        st.caption("Same unseen test photos, same six classes, so this is a fair comparison.")
        names = {"all": "All sources", "trashnet": "TrashNet (clean studio photos)",
                 "garbagev2": "Garbage Classification v2 (web photos)", "realwaste": "RealWaste (real landfill photos)"}
        st.dataframe(
            [{"Test set": names[key], "Photos": value["images"], "Old (MobileNetV2)": f"{value['old_baseline_accuracy']:.1%}",
              "New (MobileNetV3)": f"{value['new_model_accuracy']:.1%}"} for key, value in comparison.items()],
            hide_index=True, width="stretch")

        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("#### Score per class")
            st.bar_chart({WASTE_GUIDE[name][0]: clean["per_class"][name]["f1-score"] for name in metrics["classes"]},
                         horizontal=True, x_label="F1 score", color="#1f9d55")
        with right:
            st.markdown("#### Where it gets confused")
            if CONFUSION.is_file():
                st.image(str(CONFUSION), width="stretch")

with data_tab:
    st.write("")
    st.markdown("#### Data sources")
    st.markdown(
        "- **TrashNet** (Thung & Yang, 2016): 2,527 studio photos, used in the original project.\n"
        "- **RealWaste** (Single et al., 2023, CC BY 4.0): 4,752 photos taken at a real landfill facility.\n"
        "- **Garbage Classification v2** (Kunwar, MIT): 19,762 household waste photos.\n\n"
        "After removing 3,428 duplicate photos and capping the oversized clothing group, 19,358 photos remain."
    )
    st.markdown("#### What this model cannot do")
    st.markdown(
        "- It looks at one item at a time. Mixed piles and several objects in one photo confuse it.\n"
        "- It cannot see inside a container or tell whether an item is clean.\n"
        "- It only knows nine waste types. Anything else is forced into one of them, though low scores trigger *Not sure*.\n"
        "- Disposal advice is general guidance. Local collection rules decide the final answer."
    )
