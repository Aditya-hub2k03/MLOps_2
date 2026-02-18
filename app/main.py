from fastapi import FastAPI, UploadFile, File
import uvicorn
import io
import time
import logging

from app.model_loader import load_model
from app.utils import preprocess_image
from app.logging_config import setup_logging

setup_logging()

app = FastAPI(
    title="Cats vs Dogs Classifier API",
    description="Keras 3 Compatible Image Classification Service",
    version="2.0"
)

model = load_model()
request_count = 0

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "keras_version": "3.x"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global request_count
    request_count += 1

    start_time = time.time()
    logging.info(f"Request #{request_count} received")

    image_bytes = io.BytesIO(await file.read())
    processed = preprocess_image(image_bytes)

    prediction = model.predict(processed)[0][0]

    label = "Dog" if prediction > 0.5 else "Cat"
    confidence = float(prediction if prediction > 0.5 else 1 - prediction)

    latency = time.time() - start_time

    logging.info(
        f"Prediction: {label} | Confidence: {confidence:.4f} | Latency: {latency:.4f}s"
    )

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "latency_seconds": round(latency, 4),
        "total_requests": request_count
    }

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
    #uvicorn.run("app", host="0.0.0.0", port=8000)
