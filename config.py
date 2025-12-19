# config.py
# Konfigurasi dasar untuk aplikasi Streamlit klasifikasi hewan (model terbaru)

from pathlib import Path

# Ukuran input model saat training
IMG_SIZE = (224, 224)
IMG_SHAPE = (224, 224, 3)

# Lokasi folder project & model
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# Nama file model yang diletakkan di folder `models/`
# Catatan:
# - CNN end-to-end disarankan pakai .keras (paling stabil untuk load keras).
# - RandomForest tetap .joblib.
MODEL_FILES = {
    # End-to-end CNN (baseline)
    "mobilenetv3": "mobilenet_best.keras",
    "resnet50": "resnet50.joblib",          # boleh diganti ke "resnet50.keras" jika kamu punya
    "efficientnetb0": "efficientnetb0.joblib",  # boleh diganti ke "efficientnetb0.keras" jika kamu punya

    # Hybrid (MobileNetV3 features + RF)
    "mobilenetv3_rf": "rf_on_mobilenetv3_features.joblib",
}

# Daftar label kelas (harus konsisten dengan urutan saat training)
CLASS_NAMES = [
    "antelope_duiker",
    "bird",
    "blank",
    "civet_genet",
    "hog",
    "leopard",
    "monkey_prosimian",
    "rodent",
]
