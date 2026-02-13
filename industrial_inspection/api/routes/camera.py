"""Routes pour la gestion de la caméra."""

import logging
import time

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from industrial_inspection.api.dependencies import get_camera, get_detection_config
from industrial_inspection.camera.controller import CameraController
from industrial_inspection.config.detection_config import DetectionConfig
from industrial_inspection.contracts.camera import CameraStatus, CaptureResult
from industrial_inspection.image_processing.geometry import compute_pca_and_centroid
from industrial_inspection.image_processing.preprocessing import (
    array_to_base64,
    validate_image_array,
)
from industrial_inspection.image_processing.segmentation import (
    keep_largest_contour,
    segment_piece,
)
from industrial_inspection.image_processing.weighting import apply_sigmoid_weighting
from industrial_inspection.quality_checks.checker import QualityChecker

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/camera", tags=["Camera"])

# État global de la référence (à remplacer par un service si nécessaire)
reference_image: np.ndarray | None = None
reference_mask: np.ndarray | None = None
reference_computed: bool = False


@router.get("/status", response_model=CameraStatus)
async def get_status(camera: CameraController = Depends(get_camera)):
    """Obtient le statut de la caméra."""
    try:
        info = camera.get_config_info()
        return CameraStatus(
            is_initialized=camera.is_initialized,
            camera_model=info["camera_model"],
            resolution=info["resolution"],
            error=None,
        )
    except Exception as e:
        logger.error(f"Error getting camera status: {e}")
        return CameraStatus(
            is_initialized=False,
            camera_model="Unknown",
            resolution="Unknown",
            error=str(e),
        )


@router.post("/capture", response_model=CaptureResult)
async def capture_image(camera: CameraController = Depends(get_camera)):
    """Capture une image depuis la caméra."""
    try:
        start_time = time.time()

        frame = camera.capture_image()
        if frame is None:
            raise HTTPException(status_code=500, detail="Failed to capture image")

        validate_image_array(frame, np.dtype(np.uint8), 3, "captured_frame")

        image_b64 = array_to_base64(frame, format="jpeg", quality=95)

        actual_width = int(camera.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(camera.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        expected_width = camera.config.resolution.width
        expected_height = camera.config.resolution.height

        resolution_match = actual_width == expected_width and actual_height == expected_height

        total_time = (time.time() - start_time) * 1000

        logger.info(f"Image captured: {actual_width}×{actual_height} in {total_time:.2f}ms")

        return CaptureResult(
            success=True,
            image_base64=image_b64,
            width=actual_width,
            height=actual_height,
            resolution_match=resolution_match,
            error=None,
        )

    except Exception as e:
        logger.error(f"Error capturing image: {e}")
        return CaptureResult(success=False, error=str(e))


@router.post("/capture_with_mask")
async def capture_with_mask(
    camera: CameraController = Depends(get_camera),
    config: DetectionConfig = Depends(get_detection_config),
):
    """Capture une image avec génération du masque de segmentation pondéré."""
    try:
        start_time = time.time()

        frame = camera.capture_image()
        if frame is None:
            raise HTTPException(status_code=500, detail="Failed to capture image")

        validate_image_array(frame, np.dtype(np.uint8), 3, "captured_frame")

        # Segmentation
        mask_binary = segment_piece(frame, config)
        mask_binary = keep_largest_contour(mask_binary)
        mask_weighted = apply_sigmoid_weighting(mask_binary, config)

        # Encodage
        img_data_url = array_to_base64(frame.astype(np.uint8), format="jpeg", quality=95)
        mask_uint8 = (mask_weighted * 255).astype(np.uint8)
        mask_bgr = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2BGR)
        mask_data_url = array_to_base64(mask_bgr.astype(np.uint8), format="jpeg", quality=100)
        total_time = (time.time() - start_time) * 1000

        logger.info(f"Capture with mask completed in {total_time:.2f}ms")

        return {
            "success": True,
            "image": img_data_url,
            "mask": mask_data_url,
            "width": frame.shape[1],
            "height": frame.shape[0],
            "timing_ms": total_time,
        }

    except Exception as err:
        logger.error(f"Error in capture_with_mask: {err}")
        raise HTTPException(status_code=500, detail=str(err)) from err


@router.post("/set_reference")
async def set_reference_image(
    camera: CameraController = Depends(get_camera),
    config: DetectionConfig = Depends(get_detection_config),
):
    """Capture et définit l'image de référence pour les checks de qualité."""
    global reference_image, reference_mask, reference_computed

    try:
        frame = camera.capture_image()
        if frame is None:
            raise HTTPException(status_code=500, detail="Failed to capture image")

        mask = segment_piece(frame, config)
        mask = keep_largest_contour(mask)

        centroid, axes, _ = compute_pca_and_centroid(mask)

        if centroid is None or axes is None:
            raise HTTPException(status_code=400, detail="Segmentation failed - No part detected")

        reference_image = frame.copy()
        reference_mask = mask.copy()
        reference_computed = True

        angle_axis1 = np.degrees(np.arctan2(axes[0, 1], axes[0, 0]))

        logger.info(f"Reference image set: centroid={centroid}, angle={angle_axis1:.2f}°")

        return {
            "success": True,
            "message": "Reference image captured and analyzed",
            "centroid": {"x": float(centroid[0]), "y": float(centroid[1])},
            "angle_axis1": float(angle_axis1),
            "mask_area": int(np.sum(mask > 0)),
        }

    except Exception as err:
        logger.error(f"Error setting reference: {err}")
        raise HTTPException(status_code=500, detail=str(err)) from err


@router.get("/reference_status")
async def get_reference_status():
    """Vérifie si une image de référence est définie."""
    return {
        "reference_set": reference_computed,
        "message": "Reference set" if reference_computed else "No reference set",
    }


@router.post("/check_quality")
async def check_quality(
    camera: CameraController = Depends(get_camera),
    config: DetectionConfig = Depends(get_detection_config),
):
    """Capture une image et effectue tous les checks de qualité."""
    global reference_image, reference_mask, reference_computed

    if not reference_computed or reference_image is None or reference_mask is None:
        raise HTTPException(
            status_code=400,
            detail="No reference image set. Use /camera/set_reference first.",
        )

    try:
        start_time = time.time()

        test_frame = camera.capture_image()
        if test_frame is None:
            raise HTTPException(status_code=500, detail="Failed to capture image")

        test_mask = segment_piece(test_frame, config)
        test_mask = keep_largest_contour(test_mask)

        # Exécuter tous les checks
        checker = QualityChecker(config)
        report = checker.run_all_checks(reference_image, test_frame, reference_mask, test_mask)

        total_time = (time.time() - start_time) * 1000

        logger.info(f"Quality checks completed in {total_time:.2f}ms: {report.summary}")

        return {
            "success": True,
            "all_checks_passed": report.all_passed,
            "status": "ready" if report.all_passed else "not_ready",
            "checks": [check.model_dump() for check in report.checks],
            "summary": report.summary,
            "total_time_ms": total_time,
            "can_run_inference": report.all_passed,
        }

    except Exception as err:
        logger.error(f"Error in quality checks: {err}")
        raise HTTPException(status_code=500, detail=str(err)) from err


@router.post("/test")
async def test_camera(camera: CameraController = Depends(get_camera)):
    """Test rapide de la caméra."""
    try:
        start_time = time.time()

        frame = camera.capture_image()
        if frame is None:
            return {"success": False, "message": "Capture failed"}

        total_time = (time.time() - start_time) * 1000

        return {
            "success": True,
            "message": "Camera functional",
            "resolution": f"{frame.shape[1]}×{frame.shape[0]}",
            "config": camera.get_config_info(),
            "timing_ms": total_time,
        }

    except Exception as err:
        logger.error(f"Error testing camera: {err}")
        raise HTTPException(status_code=500, detail=str(err)) from err
