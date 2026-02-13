"""Analyse géométrique des masques."""


import numpy as np
import numpy.typing as npt
from sklearn.decomposition import PCA


def compute_pca_and_centroid(
    mask: npt.NDArray[np.uint8],
) -> tuple[
    tuple[float, float] | None,
    npt.NDArray[np.float64] | None,
    npt.NDArray[np.float64] | None,
]:
    y_coords, x_coords = np.where(mask > 0)

    if len(x_coords) == 0:
        return None, None, None

    centroid = (float(np.mean(x_coords)), float(np.mean(y_coords)))

    points = np.column_stack([x_coords, y_coords])
    pca = PCA(n_components=2)
    pca.fit(points)

    return (
        centroid,
        pca.components_.astype(np.float64),
        pca.explained_variance_.astype(np.float64),
    )
