# app.py
# Aplikasi Streamlit untuk mencoba model klasifikasi hewan (model terbaru)

import streamlit as st
import pandas as pd

from config import CLASS_NAMES, MODEL_FILES
from models import predict_image

st.set_page_config(
    page_title="Klasifikasi Hewan Kamera Trap",
    layout="centered",
)

st.title("📷 Klasifikasi Hewan dari Kamera Trap")

st.markdown(
    """
Upload satu gambar dari kamera trap (misalnya terang, buram, kosong, atau ada hewan).
Aplikasi akan melakukan resize 224×224, lalu menjalankan model yang kamu pilih.
"""
)

st.markdown("**File model yang diharapkan ada di folder `models/`:**")
st.code("\n".join(f"- {name}" for name in MODEL_FILES.values()), language="markdown")

st.write("---")

# ------------------------------
# Pilih model
# ------------------------------
MODEL_LABEL = {
    "mobilenetv3": "MobileNetV3Large (end-to-end)",
    "resnet50": "ResNet50 (end-to-end)",
    "efficientnetb0": "EfficientNetB0 (end-to-end)",
    "mobilenetv3_rf": "MobileNetV3Large + RandomForest (hybrid)",
}

available_model_keys = [k for k in MODEL_LABEL.keys() if k in MODEL_FILES]

model_key = st.selectbox(
    "Pilih model yang ingin diuji:",
    options=available_model_keys,
    format_func=lambda k: MODEL_LABEL[k],
)

st.write("---")

# ------------------------------
# Upload gambar
# ------------------------------
uploaded = st.file_uploader(
    "Upload gambar (.jpg / .jpeg / .png)",
    type=["jpg", "jpeg", "png"],
)

if uploaded is None:
    st.info("Silakan upload gambar terlebih dahulu.")
    st.stop()

st.image(uploaded, caption="Gambar input", use_column_width=True)

if st.button("🔍 Prediksi"):
    with st.spinner("Memproses gambar dan menjalankan model..."):
        label_str, label_idx, proba, _img_np = predict_image(uploaded, model_key)

    st.success("Prediksi selesai!")

    st.subheader("Hasil Prediksi")
    st.markdown(f"**Label:** {label_str}")
    st.markdown(f"**Index kelas:** `{label_idx}`")

    if proba is None:
        st.info("Model ini tidak menyediakan probabilitas per kelas.")
        st.stop()

    # Tabel probabilitas
    rows = []
    for i, p in enumerate(proba):
        name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class {i}"
        rows.append({"Kelas": name, "Probabilitas": float(p)})

    df = pd.DataFrame(rows).sort_values("Probabilitas", ascending=False)

    st.subheader("Probabilitas per Kelas")
    st.dataframe(df, use_container_width=True)

    st.bar_chart(df.set_index("Kelas")["Probabilitas"])
