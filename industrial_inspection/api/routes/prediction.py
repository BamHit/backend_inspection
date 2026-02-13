"""Routes pour les prédictions."""

import logging
import time

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

from industrial_inspection.contracts.prediction import PredictionResult
from industrial_inspection.image_processing.preprocessing import (
    base64_to_array,
    validate_image_array,
)
from industrial_inspection.ml.predictor import predict

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post("/", response_model=PredictionResult)
async def predict_from_upload(file: UploadFile = File(...)):
    """Effectue une prédiction sur une image uploadée."""
    try:
        start_time = time.time()

        image_bytes = await file.read()
        img_array = np.frombuffer(image_bytes, dtype=np.uint8)
        import cv2

        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(
                status_code=400, detail="Failed to decode image. Unsupported format."
            )

        validate_image_array(image, np.uint8, 3, "uploaded_image")

        result = predict(image)

        total_time = (time.time() - start_time) * 1000
        logger.info(f"Prediction completed in {total_time:.2f}ms: {result.predicted_class}")

        return result

    except Exception as err:
        logger.error(f"Error during prediction: {err}")
        raise HTTPException(status_code=500, detail=str(err)) from err


@router.post("/base64", response_model=PredictionResult)
async def predict_from_base64(data: dict):
    """Effectue une prédiction sur une image en base64.

    Body JSON:
    {
        "image": "data:image/png;base64,..."
    }
    """
    try:
        start_time = time.time()

        base64_str = data.get("image", "")
        if not base64_str:
            raise HTTPException(status_code=400, detail="No image provided")

        image = base64_to_array(base64_str, validate=True)

        result = predict(image)

        total_time = (time.time() - start_time) * 1000
        logger.info(f"Prediction from base64 completed in {total_time:.2f}ms")

        return result

    except ValueError as err:
        logger.error(f"Validation error: {err}")
        raise HTTPException(status_code=400, detail=str(err)) from err

    except Exception as err:
        logger.error(f"Error: {err}")
        raise HTTPException(status_code=500, detail=str(err)) from err


@router.post("/roi")
async def predict_roi(data: dict):
    """Prédiction sur une ROI spécifique dans une image."""
    try:
        base64_str = data.get("image", "")
        roi = data.get("roi", {})

        if not base64_str or not roi:
            raise HTTPException(status_code=400, detail="Missing image or ROI")

        image = base64_to_array(base64_str, validate=True)

        x = roi.get("x", 0)
        y = roi.get("y", 0)
        size = roi.get("size", 0)

        if x < 0 or y < 0 or size <= 0:
            raise HTTPException(status_code=400, detail="Invalid ROI coordinates")

        # Extraire la ROI
        roi_image = image[y : y + size, x : x + size]

        if roi_image.size == 0:
            raise HTTPException(status_code=400, detail="ROI out of bounds")

        result = predict(roi_image)

        return {
            "roi": roi,
            "prediction": result.model_dump(),
        }

    except Exception as err:
        logger.error(f"Error predicting ROI: {err}")
        raise HTTPException(status_code=500, detail=str(err)) from err


@router.post("/batch_rois")
async def predict_batch_rois(data: dict):
    """Prédiction sur plusieurs ROIs d'une même image."""
    try:
        start_time = time.time()

        base64_str = data.get("image", "")
        rois = data.get("rois", [])

        if not base64_str or not rois:
            raise HTTPException(status_code=400, detail="Missing image or ROIs")

        image = base64_to_array(base64_str, validate=True)

        predictions = []

        for idx, roi in enumerate(rois):
            x = roi.get("x", 0)
            y = roi.get("y", 0)
            size = roi.get("size", 0)

            roi_image = image[y : y + size, x : x + size]

            if roi_image.size == 0:
                logger.warning(f"ROI {idx} out of bounds, skipping")
                continue

            result = predict(roi_image)

            predictions.append(
                {
                    "roi_index": idx,
                    "roi": roi,
                    "prediction": result.model_dump(),
                }
            )

        total_time = (time.time() - start_time) * 1000

        logger.info(f"Batch prediction on {len(predictions)} ROIs completed in {total_time:.2f}ms")

        return {
            "total_rois": len(rois),
            "successful_predictions": len(predictions),
            "predictions": predictions,
            "timing_ms": total_time,
        }

    except Exception as err:
        logger.error(f"Error in batch prediction: {err}")
        raise HTTPException(status_code=500, detail=str(err)) from err
