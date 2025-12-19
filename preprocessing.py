# preprocessing.py
# Baca gambar + resize. Normalisasi/preprocess_input dilakukan di models.py sesuai backbone.

from typing import Tuple
import numpy as np
from PIL import Image

from config import IMG_SIZE


def load_image(file) -> Image.Image:
    """
    Menerima:
    - UploadedFile Streamlit
    - path string
    - file-like object
    - PIL.Image
    """
    if isinstance(file, Image.Image):
        img = file
    else:
        img = Image.open(file)
    return img.convert("RGB")


def preprocess_image(file) -> Tuple[np.ndarray, Image.Image]:
    """
    Return:
    - arr: np.ndarray float32 shape (1, H, W, 3) dengan range 0..255 (belum preprocess_input)
    - img_resized: PIL.Image untuk ditampilkan di Streamlit
    """
    img = load_image(file)
    img_resized = img.resize(IMG_SIZE)

    arr = np.asarray(img_resized).astype("float32")  # 0..255
    arr = np.expand_dims(arr, axis=0)                # (1,H,W,3)

    return arr, img_resized
