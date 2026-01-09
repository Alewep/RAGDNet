from abc import ABC, abstractmethod

import cv2
import networkx as nx
import numpy as np


BLACK = 1
WHITE = 0


NUM_CHANNELS_RGB = 3
NUM_CHANNELS_GRAY = 1
NDIM_GRAY = 2
BINARY_WHITE_THRESHOLD = 255

DIM_GRAY_IMAGE = 2

BINARY_MEAN_THRESHOLD = 0.5

# 8-connected neighborhood kernel
KERNEL = np.array(
    [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ]
)


def binarize(
    data: np.ndarray,
    threshold: int = 254,
) -> np.ndarray:
    img = np.asarray(data)

    # Explicitly reject floats
    if np.issubdtype(img.dtype, np.floating):
        raise TypeError("Float images are not supported.")

    # Convert to grayscale if needed
    if img.ndim != NDIM_GRAY:
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            raise ValueError("Failed to convert image to grayscale.") from e

    # Ensure uint8 for consistency
    if img.dtype != np.uint8:
        img = img.astype(np.uint8, copy=False)

    # If already binary, keep as-is
    if np.all(np.isin(img, (0, 1))):
        b = img.astype(bool, copy=False)
    else:
        b = img <= threshold

    # Make the majority class the background (0)
    if b.mean() > BINARY_MEAN_THRESHOLD:
        b = ~b

    return b.astype(np.uint8)  # {0, 1}


class GraphBuilderStrategy[T](ABC):
    def __call__(self, data: T) -> nx.Graph:
        graph = self._build_graph(data)
        # verif
        assert all(u != v for u, v in graph.edges()), "Self-loop detected"

        return graph

    @abstractmethod
    def _build_graph(self, data: T) -> nx.Graph:
        """
        Build a graph from the provided data (implemented in subclasses).

        Args:
            data: Any kind of input the concrete strategy expects.

        Returns:
            nx.Graph: The constructed graph.
        """


class LabelerStrategy(ABC):
    def __call__(self, graph: nx.Graph, colored_img: np.ndarray) -> nx.Graph:
        # 1. Check that the image has 3 channels
        if (
            colored_img.ndim != NUM_CHANNELS_RGB
            or colored_img.shape[2] != NUM_CHANNELS_RGB
        ):
            raise ValueError(
                f"Expected an RGB image (3 channels), got shape={colored_img.shape}"
            )

        # 2. Check that the image is not empty
        if colored_img.size == 0:
            raise ValueError("Empty image: no pixels to process")

        # 3. Check that the image is not grayscale (R=G=B everywhere)
        if np.all(colored_img[..., 0] == colored_img[..., 1]) and np.all(
            colored_img[..., 0] == colored_img[..., 2]
        ):
            raise ValueError(
                "Grayscale image detected: all pixels have R=G=B. "
                "Please provide a color image."
            )

        return self._label(graph, colored_img)

    @abstractmethod
    def _label(self, graph: nx.Graph, colored_img: np.ndarray) -> nx.Graph:
        """
        Abstract method for labeling a graph.

        Args:
            graph (nx.Graph): The graph to label.
            colored_img (np.ndarray): The RGB image of the colorized DXF drawing.

        Returns
        -------
            nx.Graph: The labeled graph.
        """


class SegmentStrategy(ABC):
    def __call__(self, data: np.ndarray) -> np.ndarray:
        binary = binarize(data)
        return self._segment(binary)

    @abstractmethod
    def _segment(self, binary: np.ndarray) -> np.ndarray:
        """
        Define the segmentation logic.

        Parameters
        ----------
        data : np.ndarray
            Input skeleton.

        Returns
        -------
        np.ndarray
            Segmented skeleton.
        """


class WatershedStrategy(ABC):
    def __call__(self, segmented: np.ndarray, image: np.ndarray) -> np.ndarray:
        binary = binarize(image)
        return self._watershed(segmented, binary)

    @abstractmethod
    def _watershed(self, segmented: np.ndarray, binary: np.ndarray) -> np.ndarray:
        """
        Define the watershed flooding logic.

        Parameters
        ----------
        segmented : np.ndarray
            Preprocessed skeleton.
        image : np.ndarray
            Input image.

        Returns
        -------
        np.ndarray
            Flooded skeleton mask.
        """
