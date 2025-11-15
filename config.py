# config.py
# Konfigurasi dasar untuk aplikasi Streamlit klasifikasi hewan

from pathlib import Path

# Ukuran gambar saat training (lihat di notebook: IMG_SIZE = (224, 224))
IMG_SIZE = (224, 224)
IMG_SHAPE = (224, 224, 3)

# Lokasi folder project & model
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# Nama file model .joblib (letakkan file2 ini di folder `models/`)
MODEL_FILES = {
    "best_detection": "best_detection_model_from_resnet.joblib",
    "resnet50_rf": "resnet50-rf.joblib",
    "resnet50v2_rf": "resnet50v2-rf.joblib",
    "efficientnet_rf": "efficientnet-rf.joblib",
}

# Daftar label hewan (diambil dari train_labels.csv: semua kolom selain 'id')
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
