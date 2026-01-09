import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed
import matplotlib.pyplot as plt
from ragdnet.pipelines.image_to_graph.interfaces import WatershedStrategy

KERNEL = np.array(
    [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ]
)


class DistanceTransformWatershed(WatershedStrategy):
    """
    Watershed flooding using the distance transform.
    Designed for images with white background and black skeleton/objects.
    """

    def _watershed(self, segmented: np.ndarray, binary: np.ndarray) -> np.ndarray:
        """
        Flood the segmented skeleton using distance transform and watershed.

        Parameters
        ----------
        segmented : np.ndarray
            Preprocessed skeleton (binary mask, same polarity as image).
        image : np.ndarray
            Input image (white background, black objects).

        Returns
        -------
        np.ndarray
            Label image of watershed regions grown from the skeleton.
        """

        # Distance transform inside black regions
        distance = ndi.distance_transform_bf(binary, metric="chessboard")

        # Markers from skeleton
        markers, _ = ndi.label(segmented, structure=KERNEL)

        # Watershed flooding
        return watershed(-distance, markers, mask=binary, connectivity=2)
