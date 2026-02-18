import keras
import logging

MODEL_PATH = "model.keras"

def load_model():
    logging.info("Loading Keras 3 model...")
    model = keras.models.load_model(MODEL_PATH, compile=False)
    logging.info("Model loaded successfully")
    return model
