import io
from PIL import Image
import numpy as np
from app.utils import preprocess_image

def test_preprocess_image_shape():
    # Create dummy image
    image = Image.new("RGB", (300, 300), color="white")
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes.seek(0)

    processed = preprocess_image(image_bytes)

    assert processed.shape == (1, 224, 224, 3)
    assert np.max(processed) <= 1.0
