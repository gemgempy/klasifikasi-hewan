# app.py
# Aplikasi Streamlit untuk mencoba model klasifikasi hewan

import streamlit as st

from config import CLASS_NAMES, MODEL_FILES
from models import predict_image


st.set_page_config(
    page_title="Klasifikasi Hewan Kamera Trap",
    layout="centered",
)


st.title("📷 Klasifikasi Hewan dari Kamera Trap")

st.markdown(
    """
Upload satu gambar dari kamera trap (misalnya yang sangat terang,
buram, kosong, atau ada hewan). Aplikasi ini akan melakukan
**preprocessing** (resize 224×224, normalisasi) dan:

- Untuk model **deteksi** → prediksi *ada hewan / tidak ada hewan*.
- Untuk model **klasifikasi** → prediksi jenis hewan.
"""
)

st.markdown("**File .joblib yang diharapkan ada di folder `models/`:**")
st.code(
    "\n".join(f"- {name}" for name in MODEL_FILES.values()),
    language="markdown",
)

st.write("---")

# ------------------------------
# Pilih model
# ------------------------------

MODEL_LABEL = {
    "best_detection": "Deteksi Hewan (ResNet-based)",
    "resnet50_rf": "Klasifikasi Hewan - ResNet50 + RF",
    "resnet50v2_rf": "Klasifikasi Hewan - ResNet50V2 + RF",
    "efficientnet_rf": "Klasifikasi Hewan - EfficientNetB0 + RF",
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

if uploaded is not None:
    st.image(uploaded, caption="Gambar input", use_column_width=True)

    if st.button("🔍 Prediksi"):
        with st.spinner("Memproses gambar dan menjalankan model..."):
            label_str, label_idx, proba, img_np = predict_image(uploaded, model_key)

        st.success("Prediksi selesai!")

        st.subheader("Hasil Prediksi")
        st.markdown(f"**Label:** {label_str}")
        st.markdown(f"**Index kelas:** `{label_idx}`")

        if proba is not None:
            st.write("Probabilitas per kelas:")

            if model_key == "best_detection":
                rows = [
                    {"Kelas": "Tidak ada hewan (0)", "Probabilitas": float(proba[0])},
                    {"Kelas": "Ada hewan (1)", "Probabilitas": float(proba[1])},
                ]
                st.table(rows)
            else:
                rows = []
                for i, p in enumerate(proba):
                    if i < len(CLASS_NAMES):
                        name = CLASS_NAMES[i]
                    else:
                        name = f"Class {i}"
                    rows.append({"Kelas": name, "Probabilitas": float(p)})
                st.table(rows)
else:
    st.info("Silakan upload gambar terlebih dahulu.")
