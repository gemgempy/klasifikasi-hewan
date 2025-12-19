# models.py
# Load model CNN end-to-end (.keras/.h5/.joblib) dan RandomForest (.joblib)
# + pipeline prediksi untuk 1 gambar.
#
# Model yang didukung (tanpa DenseNet):
# - ResNet50 (end-to-end)
# - MobileNetV3Large (end-to-end)
# - EfficientNetB0 (end-to-end)
# - MobileNetV3Large features + RandomForest (hybrid)

from __future__ import annotations

from typing import Dict, Tuple, Optional
import numpy as np
import joblib
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, InputLayer, Lambda

from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenetv3_preprocess

from config import MODEL_DIR, MODEL_FILES, IMG_SHAPE, CLASS_NAMES
from preprocessing import preprocess_image


# ==============================
# CACHES
# ==============================
_cnn_models: Dict[str, object] = {}
_rf_models: Dict[str, object] = {}
_feature_extractors: Dict[str, tf.keras.Model] = {}


# ==============================
# UTILS: LOAD MODEL
# ==============================
def _load_keras_or_joblib(path):
    """
    Coba load sebagai Keras model dulu (untuk .keras/.h5/folder saved_model),
    kalau gagal baru fallback ke joblib (untuk .joblib).
    """
    # 1) keras load (aman untuk .keras, .h5, SavedModel dir)
    try:
        return tf.keras.models.load_model(str(path))
    except Exception:
        pass

    # 2) joblib load
    return joblib.load(str(path))


def get_cnn_model(key: str):
    """
    Lazy-load model CNN end-to-end dari MODEL_FILES.
    key contoh: 'resnet50', 'mobilenetv3', 'efficientnetb0'
    """
    if key in _cnn_models:
        return _cnn_models[key]

    if key not in MODEL_FILES:
        raise KeyError(f"Tidak ada entri MODEL_FILES untuk key='{key}'.")

    path = MODEL_DIR / MODEL_FILES[key]
    if not path.exists():
        raise FileNotFoundError(f"File model tidak ditemukan: {path}")

    model = _load_keras_or_joblib(path)
    _cnn_models[key] = model
    return model


def get_rf_model(key: str):
    """
    Lazy-load model RandomForest dari MODEL_FILES.
    key contoh: 'mobilenetv3_rf'
    """
    if key in _rf_models:
        return _rf_models[key]

    if key not in MODEL_FILES:
        raise KeyError(f"Tidak ada entri MODEL_FILES untuk key='{key}'.")

    path = MODEL_DIR / MODEL_FILES[key]
    if not path.exists():
        raise FileNotFoundError(f"File RF tidak ditemukan: {path}")

    model = joblib.load(str(path))
    _rf_models[key] = model
    return model


# ==============================
# PREPROCESS BACKBONE
# ==============================
def _ensure_0_255_float(x: np.ndarray) -> np.ndarray:
    """
    Training kamu pakai preprocess_input pada float32 (range umumnya 0..255).
    Kalau preprocess_image ternyata mengeluarkan 0..1, kita scale balik ke 0..255.
    """
    x = x.astype("float32")
    mx = float(np.max(x)) if x.size else 0.0
    if mx <= 1.5:  # indikasi kuat datanya 0..1
        x = x * 255.0
    return x


def _preprocess_for_backbone(x: np.ndarray, backbone: str) -> np.ndarray:
    """
    x: (1,H,W,3) float32 RGB (0..255)
    backbone: 'resnet50' | 'mobilenetv3' | 'efficientnetb0'
    """
    x = _ensure_0_255_float(x)

    if backbone == "resnet50":
        return resnet_preprocess(x.copy())
    if backbone == "mobilenetv3":
        return mobilenetv3_preprocess(x.copy())
    if backbone == "efficientnetb0":
        return effnet_preprocess(x.copy())

    raise ValueError(f"Backbone '{backbone}' tidak dikenal.")


# ==============================
# FEATURE EXTRACTOR: MobileNetV3 backbone from trained model
# ==============================
def _find_backbone_mobilenet(full_model: Model) -> Optional[Model]:
    """
    Ambil backbone MobileNetV3Large dari model end-to-end yang sudah dilatih.
    Ini lebih benar dibanding bikin backbone baru dari ImageNet, karena kamu sudah fine-tune.
    """
    # kandidat: layer yang merupakan sub-Model dan namanya mengandung mobilenet
    for layer in full_model.layers:
        if isinstance(layer, tf.keras.Model) and "mobilenet" in layer.name.lower():
            return layer

    # fallback: ambil sub-Model paling besar
    candidates = [l for l in full_model.layers if isinstance(l, tf.keras.Model)]
    if candidates:
        return max(candidates, key=lambda m: len(m.layers))

    return None


def get_mobilenet_feature_extractor(mobilenet_e2e_key: str = "mobilenetv3") -> tf.keras.Model:
    """
    Feature extractor: preprocess_input -> backbone -> GlobalAveragePooling2D
    Output shape biasanya (N, 960) untuk MobileNetV3Large.
    """
    cache_key = f"{mobilenet_e2e_key}_feature_extractor"
    if cache_key in _feature_extractors:
        return _feature_extractors[cache_key]

    mobilenet_model = get_cnn_model(mobilenet_e2e_key)
    if not isinstance(mobilenet_model, tf.keras.Model):
        raise TypeError("Model MobileNet end-to-end yang diload bukan tf.keras.Model. Pastikan file benar.")

    backbone = _find_backbone_mobilenet(mobilenet_model)
    if backbone is None:
        raise RuntimeError("Backbone MobileNetV3 tidak ditemukan dari model end-to-end. Cek struktur model yang disimpan.")

    feat_model = tf.keras.Sequential(
        [
            InputLayer(input_shape=IMG_SHAPE),
            Lambda(mobilenetv3_preprocess),
            backbone,
            GlobalAveragePooling2D(),
        ],
        name="mobilenetv3_feature_extractor",
    )

    _feature_extractors[cache_key] = feat_model
    return feat_model


# ==============================
# PREDICTION HELPERS
# ==============================
def _probs_from_model_output(pred: np.ndarray) -> np.ndarray:
    """
    Beberapa model sudah output softmax, sebagian output logits.
    Kalau tidak sum=1, kita softmax-kan.
    """
    pred = np.asarray(pred)
    if pred.ndim != 2:
        pred = np.reshape(pred, (pred.shape[0], -1))

    row_sum = pred.sum(axis=1, keepdims=True)
    if np.allclose(row_sum, 1.0, atol=1e-3):
        return pred

    return tf.nn.softmax(pred, axis=1).numpy()


# ==============================
# MAIN PIPELINE: predict_image
# ==============================
def predict_image(file, model_key: str) -> Tuple[str, int, np.ndarray, np.ndarray]:
    """
    Return:
      - label_str: nama kelas
      - label_idx: index kelas
      - proba: np.ndarray shape (num_classes,) atau None
      - img_np: array gambar (H,W,3) untuk visualisasi Streamlit
    """

    # 1) Baca & resize dari preprocessing.py
    img_array, img_pil = preprocess_image(file)   # img_array biasanya (1,224,224,3)
    img_np = np.asarray(img_pil)

    # ---- CASE A: END-TO-END CNN ----
    # Harap MODEL_FILES punya key: 'resnet50', 'mobilenetv3', 'efficientnetb0'
    if model_key in ("resnet50", "mobilenetv3", "efficientnetb0"):
        model = get_cnn_model(model_key)
        if not isinstance(model, tf.keras.Model):
            raise TypeError(f"Model '{model_key}' bukan tf.keras.Model. Pastikan format file model benar.")

        x_pp = _preprocess_for_backbone(img_array, backbone=model_key)
        pred = model.predict(x_pp, verbose=0)
        probs = _probs_from_model_output(pred)[0]

        y_pred_idx = int(np.argmax(probs))
        label_str = CLASS_NAMES[y_pred_idx] if 0 <= y_pred_idx < len(CLASS_NAMES) else f"Class {y_pred_idx}"
        return label_str, y_pred_idx, probs, img_np

    # ---- CASE B: HYBRID MobileNetV3 features + RF ----
    # Harap MODEL_FILES punya key RF: 'mobilenetv3_rf'
    if model_key == "mobilenetv3_rf":
        rf_model = get_rf_model(model_key)

        feat_extractor = get_mobilenet_feature_extractor(mobilenet_e2e_key="mobilenetv3")
        feats = feat_extractor(img_array, training=False).numpy()  # (1, C)

        # prediksi RF
        y_pred_idx = int(rf_model.predict(feats)[0])

        # probabilitas RF (kalau ada)
        proba = None
        if hasattr(rf_model, "predict_proba"):
            proba_rf = rf_model.predict_proba(feats)[0]
            # map sesuai rf_model.classes_
            proba = np.zeros((len(CLASS_NAMES),), dtype="float32")
            for cls_idx, p in zip(rf_model.classes_, proba_rf):
                if 0 <= int(cls_idx) < len(CLASS_NAMES):
                    proba[int(cls_idx)] = float(p)

        label_str = CLASS_NAMES[y_pred_idx] if 0 <= y_pred_idx < len(CLASS_NAMES) else f"Class {y_pred_idx}"
        return label_str, y_pred_idx, proba, img_np

    raise KeyError(
        f"model_key '{model_key}' tidak dikenali. "
        f"Gunakan: resnet50 | mobilenetv3 | efficientnetb0 | mobilenetv3_rf"
    )
