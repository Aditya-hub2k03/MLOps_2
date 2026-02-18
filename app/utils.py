from PIL import Image
import numpy as np
import logging

IMG_SIZE = 224

def preprocess_image(image_bytes):
    logging.info("Preprocessing image")

    image = Image.open(image_bytes).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array
