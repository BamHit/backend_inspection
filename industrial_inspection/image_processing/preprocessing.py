"""
Module de preprocessing unifié pour l'inspection industrielle.
Garantit un encodage cohérent entre entraînement et inférence.

Type safety avec numpy.typing pour production.
"""

import base64
import warnings
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

# Type aliases pour clarté
ImageArray = NDArray[np.uint8]  # Image en uint8 [H, W, C]
FloatArray = NDArray[np.float32]  # Image normalisée en float32

# Color space types
ColorSpace = Literal["RGB", "BGR", "GRAY"]


# ============================================================================
# VALIDATION DES IMAGES
# ============================================================================


def validate_image_array(
    image: Any,
    expected_dtype: Any = np.uint8,
    expected_channels: int | None = 3,
    name: str = "image",
) -> None:
    """
    Valide qu'un array numpy est une image valide.

    Args:
        image: Array numpy à valider
        expected_dtype: Type de données attendu (défaut: uint8)
        expected_channels: Nombre de canaux attendus (None = pas de validation)
        name: Nom de l'image pour les messages d'erreur

    Raises:
        ValueError: Si l'image n'est pas valide
        TypeError: Si le type est incorrect
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{name} doit être un numpy.ndarray, reçu: {type(image)}")

    if image.dtype != expected_dtype:
        raise ValueError(f"{name} doit être de type {expected_dtype}, reçu: {image.dtype}")

    if image.ndim != 3:
        raise ValueError(f"{name} doit avoir 3 (color) dimensions, reçu: {image.ndim} dimensions")

    if expected_channels is not None:
        if image.ndim == 3 and image.shape[2] != expected_channels:
            raise ValueError(
                f"{name} doit avoir {expected_channels} canaux, reçu: {image.shape[2]} canaux"
            )

    if image.size == 0:
        raise ValueError(f"{name} est vide (size=0)")

    # Vérifier les valeurs
    if expected_dtype == np.uint8:
        if image.min() < 0 or image.max() > 255:
            warnings.warn(
                f"{name}: valeurs hors range [0, 255]: min={image.min()}, max={image.max()}",
                stacklevel=2,
            )


def validate_image_shape(
    image: NDArray,
    min_size: int = 32,  # TODO changer les min et max ici
    max_size: int = 8192,
    name: str = "image",
) -> None:
    """
    Valide les dimensions d'une image.

    Args:
        image: Image à valider
        min_size: Taille minimale (hauteur et largeur)
        max_size: Taille maximale
        name: Nom pour les messages d'erreur

    Raises:
        ValueError: Si les dimensions sont invalides
    """
    height, width = image.shape[:2]

    if height < min_size or width < min_size:
        raise ValueError(
            f"{name}: dimensions trop petites ({height}x{width}). Minimum: {min_size}x{min_size}"
        )

    if height > max_size or width > max_size:
        raise ValueError(
            f"{name}: dimensions trop grandes ({height}x{width}). Maximum: {max_size}x{max_size}"
        )


# ============================================================================
# CHARGEMENT D'IMAGES
# ============================================================================


def load_image_cv2(
    path: str | Path, color_space: ColorSpace = "RGB", validate: bool = True
) -> ImageArray:
    """
    Charge une image avec OpenCV avec conversion de couleur garantie.

    Args:
        path: Chemin vers l'image
        color_space: Espace colorimétrique de sortie ("RGB", "BGR", "GRAY")
        validate: Si True, valide l'image chargée

    Returns:
        Image en numpy array [H, W, C] en uint8

    Raises:
        FileNotFoundError: Si le fichier n'existe pas
        ValueError: Si l'image ne peut pas être chargée ou est invalide
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image non trouvée: {path}")

    # Charger l'image (OpenCV charge en BGR par défaut)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Impossible de charger l'image: {path}")

    # Conversion d'espace colorimétrique
    if color_space == "RGB":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_space == "GRAY":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Ajouter dimension canal pour cohérence
        image = np.expand_dims(image, axis=-1)
    elif color_space == "BGR":
        pass  # Déjà en BGR
    else:
        raise ValueError(f"color_space invalide: {color_space}")

    # Validation
    if validate:
        validate_image_array(image, np.uint8, 3 if color_space != "GRAY" else 1, str(path))
        validate_image_shape(image, name=str(path))

    return image


def load_image_pil(
    path: str | Path, color_space: ColorSpace = "RGB", validate: bool = True
) -> ImageArray:
    """
    Charge une image avec PIL/Pillow.

    Args:
        path: Chemin vers l'image
        color_space: Espace colorimétrique de sortie
        validate: Si True, valide l'image chargée

    Returns:
        Image en numpy array [H, W, C] en uint8
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image non trouvée: {path}")

    try:
        pil_image = Image.open(path)

        # Conversion
        if color_space == "RGB":
            pil_image = pil_image.convert("RGB")
        elif color_space == "GRAY":
            pil_image = pil_image.convert("L")
        elif color_space == "BGR":
            pil_image = pil_image.convert("RGB")
            # PIL ne supporte pas BGR directement, on convertira après

        # Convertir en numpy
        image = np.array(pil_image, dtype=np.uint8)

        # Assurer 3 dimensions
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)

        # Conversion BGR si nécessaire
        if color_space == "BGR":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Validation
        if validate:
            validate_image_array(image, np.uint8, 3 if color_space != "GRAY" else 1, str(path))
            validate_image_shape(image, name=str(path))

        return image

    except Exception as err:
        raise ValueError(f"Erreur lors du chargement de {path}: {err}") from err


# ============================================================================
# CONVERSION D'ESPACE COLORIMÉTRIQUE
# ============================================================================


def convert_color_space(
    image: ImageArray, from_space: ColorSpace, to_space: ColorSpace, validate: bool = True
) -> ImageArray:
    """
    Convertit une image d'un espace colorimétrique à un autre.

    Args:
        image: Image en uint8 [H, W, C]
        from_space: Espace source
        to_space: Espace destination
        validate: Si True, valide l'entrée

    Returns:
        Image convertie en uint8
    """
    if validate:
        expected_channels = 1 if from_space == "GRAY" else 3
        validate_image_array(image, np.uint8, expected_channels, "input image")

    if from_space == to_space:
        return image.copy()

    # Mapping des conversions OpenCV
    conversion_map = {
        ("RGB", "BGR"): cv2.COLOR_RGB2BGR,
        ("BGR", "RGB"): cv2.COLOR_BGR2RGB,
        ("RGB", "GRAY"): cv2.COLOR_RGB2GRAY,
        ("BGR", "GRAY"): cv2.COLOR_BGR2GRAY,
        ("GRAY", "RGB"): cv2.COLOR_GRAY2RGB,
        ("GRAY", "BGR"): cv2.COLOR_GRAY2BGR,
    }

    conversion_code = conversion_map.get((from_space, to_space))

    if conversion_code is None:
        raise ValueError(f"Conversion non supportée: {from_space} -> {to_space}")

    converted = cv2.cvtColor(image, conversion_code)

    # Assurer 3 dimensions pour grayscale
    if to_space == "GRAY" and converted.ndim == 2:
        converted = np.expand_dims(converted, axis=-1)

    return converted


# ============================================================================
# NORMALISATION POUR DEEP LEARNING
# ============================================================================


def normalize_imagenet(image: ImageArray, validate: bool = True) -> FloatArray:
    """
    Normalise une image avec les statistiques ImageNet.

    Mean: [0.485, 0.456, 0.406]
    Std:  [0.229, 0.224, 0.225]

    Args:
        image: Image en RGB uint8 [H, W, 3]
        validate: Si True, valide l'entrée

    Returns:
        Image normalisée en float32 [H, W, 3]
    """
    if validate:
        validate_image_array(image, np.uint8, 3, "image")

    # Conversion en float32 et normalisation [0, 1]
    image_float = image.astype(np.float32) / 255.0

    # Statistiques ImageNet
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # Normalisation
    normalized = (image_float - mean) / std

    return normalized


def denormalize_imagenet(image: FloatArray, clip: bool = True) -> ImageArray:
    """
    Dénormalise une image normalisée avec ImageNet.

    Args:
        image: Image normalisée en float32 [H, W, 3]
        clip: Si True, clip les valeurs dans [0, 255]

    Returns:
        Image en uint8 [H, W, 3]
    """
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # Dénormalisation
    denormalized = (image * std) + mean

    # Conversion en uint8
    denormalized = denormalized * 255.0

    if clip:
        denormalized = np.clip(denormalized, 0, 255)

    return denormalized.astype(np.uint8)


# ============================================================================
# PIPELINE UNIFIÉ POUR ENTRAÎNEMENT
# ============================================================================


class ImagePreprocessor:
    """
    Classe de preprocessing unifié garantissant la cohérence entre
    entraînement et inférence.
    """

    def __init__(
        self,
        target_size: tuple[int, int] = (128, 128),
        color_space: ColorSpace = "RGB",
        normalize: bool = True,
        use_grayscale: bool = False,
    ):
        """
        Args:
            target_size: (height, width) de sortie
            color_space: Espace colorimétrique ("RGB", "BGR", "GRAY")
            normalize: Si True, applique normalisation ImageNet
            use_grayscale: Si True, convertit en grayscale
        """
        self.target_size = target_size
        self.color_space = color_space
        self.normalize = normalize
        self.use_grayscale = use_grayscale

        # Validation
        if use_grayscale and color_space != "GRAY":
            warnings.warn(
                f"use_grayscale=True mais color_space={color_space}. Forçage à GRAY.", stacklevel=2
            )
            self.color_space = "GRAY"

    def __call__(self, image: ImageArray | str | Path) -> FloatArray:
        """
        Applique le preprocessing complet.

        Args:
            image: Image (array ou chemin)

        Returns:
            Image preprocessée en float32 [H, W, C]
        """
        # Charger si chemin
        if isinstance(image, (str | Path)):
            image_to_process = load_image_cv2(image, self.color_space)
        else:
            validate_image_array(image, np.uint8, name="input")
            image_to_process = image

        # Redimensionner
        image_resized = cv2.resize(
            image_to_process,
            (self.target_size[1], self.target_size[0]),  # OpenCV attend (width, height)
            interpolation=cv2.INTER_LINEAR,
        )

        # Normaliser
        if self.normalize:
            # Conversion en RGB si nécessaire pour ImageNet
            if self.color_space != "RGB":
                image_rgb = convert_color_space(image_resized, self.color_space, "RGB")
            else:
                image_rgb = image_resized

            image_normalized = normalize_imagenet(image_rgb)
        else:
            # Juste conversion en float [0, 1]
            image_normalized = image_resized.astype(np.float32) / 255.0

        return image_normalized

    def to_tensor(self, image: ImageArray | FloatArray) -> torch.Tensor:
        """
        Convertit une image preprocessée en tensor PyTorch.

        Args:
            image: Image en float32 [H, W, C]

        Returns:
            Tensor [C, H, W]
        """
        # Transposer de [H, W, C] à [C, H, W]
        tensor = torch.from_numpy(image.transpose(2, 0, 1))
        return tensor

    def preprocess_to_tensor(self, image: ImageArray | str | Path) -> torch.Tensor:
        """
        Pipeline complet: chargement → preprocessing → tensor.

        Args:
            image: Image ou chemin

        Returns:
            Tensor prêt pour le modèle [C, H, W]
        """
        preprocessed = self(image)
        tensor = self.to_tensor(preprocessed)
        return tensor


# ============================================================================
# PIPELINE POUR INFÉRENCE (API)
# ============================================================================


def preprocess_for_inference(
    image: ImageArray,
    target_size: tuple[int, int] = (128, 128),
    from_color_space: ColorSpace = "BGR",
) -> torch.Tensor:
    """
    Preprocessing pour inférence API (depuis OpenCV/caméra).

    IMPORTANT: Cette fonction doit être STRICTEMENT IDENTIQUE
    au preprocessing d'entraînement.

    Args:
        image: Image en uint8 [H, W, C]
        target_size: (height, width)
        from_color_space: Espace source ("BGR" pour OpenCV, "RGB" pour PIL)

    Returns:
        Tensor [1, C, H, W] prêt pour le modèle
    """
    # Validation
    validate_image_array(image, np.uint8, name="inference_image")

    # Conversion en RGB (ImageNet est en RGB)
    if from_color_space != "RGB":
        image_rgb = convert_color_space(image, from_color_space, "RGB")
    else:
        image_rgb = image.copy()

    # Redimensionner
    image_resized = cv2.resize(
        image_rgb, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR
    )

    # Normalisation ImageNet
    image_normalized = normalize_imagenet(image_resized)

    # Conversion en tensor [C, H, W]
    tensor = torch.from_numpy(image_normalized.transpose(2, 0, 1))

    # Ajouter batch dimension [1, C, H, W]
    tensor = tensor.unsqueeze(0)

    return tensor


# ============================================================================
# DÉCODAGE BASE64 POUR API
# ============================================================================


def base64_to_array(base64_str: str, validate: bool = True) -> ImageArray:
    """
    Décode une image base64 en array numpy.

    Args:
        base64_str: String base64 (avec ou sans préfixe data:image)
        validate: Si True, valide l'image résultante

    Returns:
        Image en BGR uint8 [H, W, 3] (format OpenCV)

    Raises:
        ValueError: Si le décodage échoue
    """
    try:
        # Retirer le préfixe data:image si présent
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]

        # Décoder
        img_bytes = base64.b64decode(base64_str)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)

        # Décoder avec OpenCV (retourne BGR)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("cv2.imdecode a retourné None")

        # Validation
        if validate:
            validate_image_array(image, np.uint8, 3, "base64_image")

        return image

    except Exception as err:
        raise ValueError(f"Erreur décodage base64: {err}") from err


def array_to_base64(
    image: ImageArray,
    format: str = "jpeg",  # ✅ Changement: défaut JPEG au lieu de PNG
    quality: int = 95,  # ✅ AJOUT: paramètre qualité
) -> str:
    """
    Convertit une image numpy array en data URL base64.

    Args:
        image: Image BGR uint8 [H, W, 3]
        format: "jpeg" (recommandé pour photos) ou "png" (pour graphiques)
        quality: 1-100, qualité JPEG (ignoré pour PNG)

    Returns:
        Data URL base64 (e.g., "data:image/jpeg;base64,...")
    """
    validate_image_array(image, np.uint8, 3, "array_to_base64_input")

    if format.lower() == "jpeg" or format.lower() == "jpg":
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, buffer = cv2.imencode(".jpg", image, encode_param)
        mime_type = "image/jpeg"
    else:
        raise ValueError(f"Format non supporté: {format}. Utilisez 'jpeg'")

    if not success:
        raise ValueError(f"Échec de l'encodage en {format}")

    base64_str = base64.b64encode(buffer).decode("utf-8")
    data_url = f"data:{mime_type};base64,{base64_str}"

    print(f"  Encodé en {format.upper()}: {len(data_url) / 1024:.0f} KB")

    return data_url
