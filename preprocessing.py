# preprocessing.py
# Fungsi untuk load & preprocessing gambar sebelum masuk ke CNN

from typing import Tuple
import numpy as np
from PIL import Image

from config import IMG_SIZE


def load_image(file) -> Image.Image:
    """
    Menerima:
    - path string / Path
    - file-like object dari st.file_uploader
    - PIL.Image.Image

    dan mengembalikan PIL image dalam mode RGB.
    """
    if isinstance(file, Image.Image):
        img = file
    else:
        img = Image.open(file)
    return img.convert("RGB")


def preprocess_image(file) -> Tuple[np.ndarray, Image.Image]:
    """Load gambar, resize, normalisasi.

    Sama dengan ImageDataGenerator:
      rescale=(1.0/127.5)-1.0 -> range piksel [-1, 1]

    Output:
      - image_array: np.ndarray shape (1, H, W, 3)
      - img_resized: PIL.Image untuk ditampilkan di Streamlit
    """
    img = load_image(file)
    img_resized = img.resize(IMG_SIZE)

    arr = np.asarray(img_resized).astype("float32")
    arr = (arr / 127.5) - 1.0    # rescale ke [-1, 1]
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)

    return arr, img_resized
