from typing import Tuple
import numpy as np
from PIL import Image

from config import IMG_SIZE


def load_image(file) -> Image.Image:
    if isinstance(file, Image.Image):
        img = file
    else:
        img = Image.open(file)
    return img.convert("RGB")


def preprocess_image(file) -> Tuple[np.ndarray, Image.Image]:
    img = load_image(file)
    img_resized = img.resize(IMG_SIZE)

    arr = np.asarray(img_resized).astype("float32")
    arr = (arr / 127.5) - 1.0   # SAMA kayak ImageDataGenerator(rescale=(1/127.5)-1.0)
    arr = np.expand_dims(arr, axis=0)

    return arr, img_resized
