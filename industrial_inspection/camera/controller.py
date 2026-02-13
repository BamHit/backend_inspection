"""Contrôleur de caméra refactorisé."""

import logging
import time
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from industrial_inspection.config.camera_config import CameraConfig
from industrial_inspection.exceptions import CameraException

logger = logging.getLogger(__name__)


def apply_white_balance(
    frame: npt.NDArray[np.uint8],
    wb: dict[str, float],
) -> npt.NDArray[np.uint8]:
    """Applique la balance des blancs logicielle.

    Args:
        frame: Image BGR.
        wb: Coefficients de balance des blancs {'r': float, 'g': float, 'b': float}.

    Returns:
        Image avec balance des blancs appliquée.
    """
    frame_float = frame.astype(np.float32)
    frame_float[:, :, 2] *= wb["r"]  # Red
    frame_float[:, :, 1] *= wb["g"]  # Green
    frame_float[:, :, 0] *= wb["b"]  # Blue
    return np.clip(frame_float, 0, 255).astype(np.uint8)


class CameraController:
    """Contrôleur pour la caméra Supertek ST10-48MPAF.

    Gère l'initialisation et la capture avec paramètres manuels.
    Utilise la configuration Pydantic pour tous les paramètres.
    """

    def __init__(self, config: CameraConfig):
        """Initialise le contrôleur avec une configuration.

        Args:
            config: Configuration Pydantic de la caméra.
        """
        self.config = config
        self.cap: cv2.VideoCapture = cv2.VideoCapture()
        self.is_initialized = False
        self.software_wb: dict[str, float] | None = None

    def initialize(self) -> bool:
        """Initialise la caméra avec les paramètres configurés.

        ORDRE CRITIQUE:
        1. Ouvrir la caméra
        2. MJPG format
        3. Résolution
        4. FPS
        5. PAUSE (temps de stabilisation)
        6. Paramètres manuels (exposure, focus, WB, gain)
        7. Vérification résolution
        8. Frames de chauffe

        Returns:
            True si succès, False sinon.

        Raises:
            CameraException: Si l'initialisation échoue.
        """
        try:
            cam_index = self.config.cam_index
            logger.info(f"Tentative d'ouverture caméra index {cam_index}")

            # 1. Ouvrir la caméra avec DirectShow (Windows)
            self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)

            if not self.cap.isOpened():
                raise CameraException(f"Impossible d'ouvrir la caméra à l'index {cam_index}")

            logger.info("Caméra ouverte avec succès")

            # 2. Format MJPG
            fourcc = cv2.VideoWriter.fourcc(*self.config.fourcc)
            self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)

            # 3. Résolution
            width = self.config.resolution.width
            height = self.config.resolution.height
            logger.info(f"Configuration résolution: {width}×{height}")

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            # 4. FPS
            self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)

            # 5. PAUSE VITALE (stabilisation du driver)
            warmup_time = self.config.warmup_time
            logger.info(f"Stabilisation caméra ({warmup_time}s)...")
            time.sleep(warmup_time)

            # 6. Paramètres manuels

            # Manual Exposure (CRITIQUE: 0.75 pour DirectShow = mode manuel)
            if not self.config.exposure.auto:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # 0.75 = manuel sous DirectShow
                self.cap.set(cv2.CAP_PROP_EXPOSURE, self.config.exposure.value)
                logger.info(f"Exposure: {self.config.exposure.value}")

            # Manual Focus
            if not self.config.focus.auto:
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # 0 = désactivé
                self.cap.set(cv2.CAP_PROP_FOCUS, self.config.focus.value)
                logger.info(f"Focus: {self.config.focus.value}")

            # WB software (stocké pour application ultérieure)
            self.software_wb = {
                "r": self.config.white_balance.r,
                "g": self.config.white_balance.g,
                "b": self.config.white_balance.b,
            }

            # Image Processing
            proc = self.config.processing
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, proc.brightness)
            self.cap.set(cv2.CAP_PROP_CONTRAST, proc.contrast)
            self.cap.set(cv2.CAP_PROP_SATURATION, proc.saturation)
            self.cap.set(cv2.CAP_PROP_SHARPNESS, proc.sharpness)
            self.cap.set(cv2.CAP_PROP_GAMMA, proc.gamma)

            # Manual Gain
            self.cap.set(cv2.CAP_PROP_GAIN, self.config.gain)
            logger.info(f"Gain: {self.config.gain}")

            # 7. Vérification de la résolution effective
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            logger.info(f"Résolution effective: {actual_width}×{actual_height}")

            if actual_width != width or actual_height != height:
                logger.warning(
                    f"Résolution différente: attendue {width}×{height}, "
                    f"obtenue {actual_width}×{actual_height}"
                )
                # Ne pas échouer, juste avertir
                # return False

            # 8. Frames de chauffe (ignorer les premières captures)
            warmup_frames = self.config.warmup_frames
            logger.info(f"Frames de chauffe ({warmup_frames})...")
            for _ in range(warmup_frames):
                self.cap.read()

            self.is_initialized = True
            logger.info("Caméra initialisée avec succès!")

            return True

        except Exception as err:
            logger.error(f"Erreur lors de l'initialisation: {err}", exc_info=True)
            if self.cap:
                self.cap.release()
            raise CameraException(f"Échec de l'initialisation: {err}") from err

    def capture_image(self) -> npt.NDArray[np.uint8] | None:
        """Capture une image depuis la caméra.

        Tente de se reconnecter automatiquement si la caméra est déconnectée.

        Returns:
            Image numpy BGR uint8, ou None en cas d'échec.
        """
        # Vérifier l'état de la caméra
        if not self.is_initialized or self.cap is None or not self.cap.isOpened():
            logger.warning("Caméra déconnectée, tentative de reconnexion...")
            try:
                if not self.initialize():
                    logger.error("Reconnexion échouée")
                    return None
            except CameraException as e:
                logger.error(f"Reconnexion échouée: {e}")
                return None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            logger.error("Erreur lecture frame (ret=False ou frame vide)")
            return None

        frame = frame.astype(np.uint8)
        if self.software_wb is not None:
            frame = apply_white_balance(frame, self.software_wb)

        logger.debug(f"Frame capturée: {frame.shape[1]} and {frame.shape[0]}")
        return frame.astype(np.uint8)

    def release(self) -> None:
        """Libère la caméra."""
        if self.cap:
            self.cap.release()
            self.is_initialized = False
            logger.info("Caméra libérée")

    def get_config_info(self) -> dict[str, Any]:
        """Retourne les informations de configuration.

        Returns:
            Dictionnaire avec les paramètres principaux.
        """
        return {
            "camera_model": self.config.camera_model,
            "resolution": f"{self.config.resolution.width}×{self.config.resolution.height}",
            "white_balance": self.software_wb,
            "focus": self.config.focus.value,
            "exposure": self.config.exposure.value,
            "gain": self.config.gain,
            **self.config.notes.model_dump(),
        }
