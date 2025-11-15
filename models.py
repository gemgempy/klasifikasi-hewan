# models.py
# Load backbone CNN (ResNet50, ResNet50V2, EfficientNetB0)
# dan model RandomForest (.joblib), lalu sediakan fungsi prediksi.

from typing import Dict, Tuple
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, ResNet50V2, EfficientNetB0

from config import MODEL_DIR, MODEL_FILES, IMG_SHAPE, CLASS_NAMES
from preprocessing import preprocess_image


# ------------------------------
# Load backbone CNN (pretrained)
# ------------------------------

_backbones: Dict[str, tf.keras.Model] = {}
_rf_models: Dict[str, object] = {}  # sklearn RandomForest atau sejenisnya


def get_backbone(name: str) -> tf.keras.Model:
    """Lazy-load backbone CNN agar tidak memakan memori di awal."""
    if name in _backbones:
        return _backbones[name]

    if name == "resnet50":
        model = ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=IMG_SHAPE,
            pooling="avg",
        )
    elif name == "resnet50v2":
        model = ResNet50V2(
            weights="imagenet",
            include_top=False,
            input_shape=IMG_SHAPE,
            pooling="avg",
        )
    elif name == "efficientnet":
        model = EfficientNetB0(
            weights="imagenet",
            include_top=False,
            input_shape=IMG_SHAPE,
            pooling="avg",
        )
    else:
        raise ValueError(f"Backbone '{name}' tidak dikenal.")

    _backbones[name] = model
    return model


# Mapping: nama model RF -> backbone yang dipakai
BACKBONE_FOR_MODEL = {
    "best_detection": "resnet50",
    "resnet50_rf": "resnet50",
    "resnet50v2_rf": "resnet50v2",
    "efficientnet_rf": "efficientnet",
}


# ------------------------------
# Load model RandomForest (.joblib)
# ------------------------------

def get_rf_model(key: str):
    """Lazy-load RandomForest/ML model dari file .joblib."""
    if key in _rf_models:
        return _rf_models[key]

    if key not in MODEL_FILES:
        raise KeyError(f"Tidak ada entri MODEL_FILES untuk key='{key}'.")

    path = MODEL_DIR / MODEL_FILES[key]
    if not path.exists():
        raise FileNotFoundError(
            f"File model '{path}' tidak ditemukan. Pastikan file .joblib ada di folder 'models/'."
        )

    model = joblib.load(path)
    _rf_models[key] = model
    return model


# ------------------------------
# Pipeline prediksi 1 gambar
# ------------------------------

def extract_features(image_array: np.ndarray, backbone_key: str) -> np.ndarray:
    """Gunakan backbone CNN untuk mengubah gambar -> fitur vektor."""
    backbone = get_backbone(backbone_key)
    feats = backbone.predict(image_array)
    feats = feats.reshape((feats.shape[0], -1))
    return feats


def predict_image(file, model_key: str) -> Tuple[str, int, np.ndarray, np.ndarray]:
    """Pipeline lengkap:
    1. Preprocess gambar (resize, normalisasi)
    2. Ekstrak fitur dengan backbone yang sesuai
    3. Prediksi dengan model .joblib

    Return:
      - label_str: nama kelas (string)
      - label_idx: index kelas (int)
      - proba: probabilitas per kelas (np.ndarray) atau None
      - img_np: array gambar (H, W, 3) untuk visualisasi tambahan
    """
    if model_key not in BACKBONE_FOR_MODEL:
        raise KeyError(f"Model key '{model_key}' tidak dikenali.")

    # 1) Preprocess gambar
    img_array, img_pil = preprocess_image(file)
    img_np = np.asarray(img_pil)

    # 2) Ekstrak fitur
    backbone_key = BACKBONE_FOR_MODEL[model_key]
    features = extract_features(img_array, backbone_key)

    # 3) Load model RF dan prediksi
    rf_model = get_rf_model(model_key)
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
