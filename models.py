# models.py
# Load backbone CNN + RandomForest (.joblib) dan sediakan fungsi prediksi.

from typing import Dict, Tuple
import numpy as np
import joblib
import tensorflow as tf

from tensorflow.keras.applications import ResNet50, EfficientNetB0, MobileNetV3Large
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

from config import MODEL_DIR, MODEL_FILES, IMG_SHAPE, CLASS_NAMES
from preprocessing import preprocess_image

# ------------------------------
# FEATURE EXTRACTOR (256 dim)
# ------------------------------

_feature_extractors: Dict[str, tf.keras.Model] = {}
_rf_models: Dict[str, object] = {}


def build_feature_extractor(name: str) -> tf.keras.Model:
    """
    Membangun model feature extractor:
      base_model -> GAP -> Dense(256, relu)
    Output: vektor fitur panjang 256 (sesuai training RandomForest).
    """
    if name == "resnet50":
        base = ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=IMG_SHAPE,
        )
    elif name == "mobilenetv3":
        base = MobileNetV3Large(
            weights="imagenet",
            include_top=False,
            input_shape=IMG_SHAPE,
        )
    elif name == "efficientnet":
        base = EfficientNetB0(
            weights="imagenet",
            include_top=False,
            input_shape=IMG_SHAPE,
        )
    else:
        raise ValueError(f"Backbone '{name}' tidak dikenal.")

    x = base.output
    x = GlobalAveragePooling2D(name=f"{name}_gap")(x)
    feat = Dense(256, activation="relu", name=f"{name}_fc256")(x)

    extractor = Model(
        inputs=base.input,
        outputs=feat,
        name=f"{name}_feature_extractor",
    )
    return extractor


def get_feature_extractor(name: str) -> tf.keras.Model:
    """Lazy-load feature extractor supaya tidak membangun berulang-ulang."""
    if name not in _feature_extractors:
        _feature_extractors[name] = build_feature_extractor(name)
    return _feature_extractors[name]


# Mapping: model RF -> backbone
BACKBONE_FOR_MODEL = {
    "best_detection": "resnet50",
    "resnet50_rf": "resnet50",
    "mobilenetv3_rf": "mobilenetv3",   
    "efficientnet_rf": "efficientnet",
}


# ------------------------------
# LOAD MODEL RANDOMFOREST (.joblib)
# ------------------------------

def get_rf_model(key: str):
    """Lazy-load RandomForest / model ML dari file .joblib di folder models/."""
    if key in _rf_models:
        return _rf_models[key]

    if key not in MODEL_FILES:
        raise KeyError(f"Tidak ada entri MODEL_FILES untuk key='{key}'.")

    path = MODEL_DIR / MODEL_FILES[key]
    if not path.exists():
        raise FileNotFoundError(
            f"File model '{path}' tidak ditemukan. "
            f"Pastikan file .joblib ada di folder 'models/'."
        )

    model = joblib.load(path)
    _rf_models[key] = model
    return model


# ------------------------------
# PIPELINE PREDIKSI 1 GAMBAR
# ------------------------------

def extract_features(image_array: np.ndarray, backbone_key: str) -> np.ndarray:
    """
    image_array: (1, H, W, 3) hasil preprocess_image
    backbone_key: 'resnet50' / 'resnet50v2' / 'efficientnet'
    Output: fitur shape (1, 256)
    """
    extractor = get_feature_extractor(backbone_key)
    feats = extractor.predict(image_array)

    # DEBUG: kalau mau cek di log Streamlit Cloud
    print(">>> FEATURE SHAPE:", feats.shape)

    feats = feats.reshape((feats.shape[0], -1))  # (1, 256)
    return feats


def predict_image(file, model_key: str) -> Tuple[str, int, np.ndarray, np.ndarray]:
    """
    Pipeline lengkap:
    1. Preprocess gambar (resize, normalisasi)
    2. Ekstrak fitur 256-dim dengan backbone sesuai
    3. Prediksi dengan model RandomForest .joblib

    Return:
      - label_str: nama kelas (string)
      - label_idx: index kelas (int)
      - proba: probabilitas per kelas (np.ndarray) atau None
      - img_np: array gambar (H, W, 3) untuk visualisasi
    """
    if model_key not in BACKBONE_FOR_MODEL:
        raise KeyError(f"Model key '{model_key}' tidak dikenali.")

    # 1) Preprocess gambar
    img_array, img_pil = preprocess_image(file)
    img_np = np.asarray(img_pil)

    # 2) Ekstrak fitur 256-dim
    backbone_key = BACKBONE_FOR_MODEL[model_key]
    features = extract_features(img_array, backbone_key)

    # 3) Load model RF + prediksi
    rf_model = get_rf_model(model_key)
    print(">>> RF expects n_features_in_ =", getattr(rf_model, "n_features_in_", None))
    print(">>> Features shape being passed to RF:", features.shape)

    y_pred_idx = int(rf_model.predict(features)[0])

    proba = None
    if hasattr(rf_model, "predict_proba"):
        proba = rf_model.predict_proba(features)[0]

    # 4) Mapping index -> label string
    if model_key == "best_detection":
        # asumsi 0 = blank / tidak ada hewan, 1 = ada hewan
        label_str = "Ada hewan" if y_pred_idx == 1 else "Tidak ada hewan"
    else:
        if 0 <= y_pred_idx < len(CLASS_NAMES):
            label_str = CLASS_NAMES[y_pred_idx]
        else:
            label_str = f"Class {y_pred_idx}"

    return label_str, y_pred_idx, proba, img_np
