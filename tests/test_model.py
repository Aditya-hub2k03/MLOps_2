import numpy as np
from app.model_loader import load_model

def test_model_prediction_shape():
    model = load_model()

    dummy_input = np.random.rand(1, 224, 224, 3).astype("float32")

    prediction = model.predict(dummy_input)

    assert prediction.shape[0] == 1
