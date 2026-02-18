from fastapi import FastAPI, UploadFile, File
import uvicorn
import io
import time
import logging

from app.model_loader import load_model
from app.utils import preprocess_image
from app.logging_config import setup_logging
from app.performance_tracker import log_performance

setup_logging()

app = FastAPI(
    title="Cats vs Dogs Classifier API",
    version="2.0"
)

model = load_model()

# Monitoring variables
request_count = 0
total_latency = 0.0


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True
    }


@app.get("/metrics")
def metrics():
    avg_latency = (
        total_latency / request_count if request_count > 0 else 0
    )

    return {
        "total_requests": request_count,
        "average_latency_seconds": round(avg_latency, 4)
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global request_count, total_latency

    request_count += 1
    start_time = time.time()

    logging.info(f"Request #{request_count} received")

    image_bytes = io.BytesIO(await file.read())
    processed = preprocess_image(image_bytes)

    prediction = model.predict(processed)[0][0]

    label = "Dog" if prediction > 0.5 else "Cat"
    confidence = float(prediction if prediction > 0.5 else 1 - prediction)

    latency = time.time() - start_time
    total_latency += latency

    logging.info(
        f"Prediction: {label} | Confidence: {confidence:.4f} | "
        f"Latency: {latency:.4f}s"
    )

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "latency_seconds": round(latency, 4),
        "total_requests": request_count
    }


# Post-deployment performance tracking
@app.post("/feedback")
def feedback(true_label: str, predicted_label: str):
    log_performance(true_label, predicted_label)

    logging.info(
        f"Feedback logged | True: {true_label} | Predicted: {predicted_label}"
    )

    return {"message": "Feedback recorded"}
    

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
