"""Interfaces for segmentation, colorization, and rendering strategies."""

from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np
from ezdxf.document import Drawing


class SegmenterStrategy(ABC):
    """
    Abstract base class for segmenting a DXF drawing.

    This strategy provides an interface for splitting a DXF drawing into
    multiple drawings. Subclasses must implement the _segment method.
    """

    def __call__(self, draw: Drawing) -> Iterable[Drawing]:
        """Call the segmentation method on the DXF drawing."""
        return self._segment(draw)

    @abstractmethod
    def _segment(self, draw: Drawing) -> Iterable[Drawing]:
        """
        Abstract method for segmenting a DXF drawing.

        Args:
            draw (Drawing): The DXF drawing to segment.

        Returns
        -------
            List[Drawing]: A list of segmented DXF drawings.
        """


class ColorizerStrategy(ABC):
    """
    Abstract base class for colorizing a DXF drawing.

    This strategy defines an interface for colorizing a DXF drawing and converting
    it into an image array. Subclasses must implement the _colorize method.
    """

    def __call__(self, draw: Drawing) -> Drawing:
        """Call the colorization method on the DXF drawing."""
        return self._colorize(draw)

    @abstractmethod
    def _colorize(self, draw: Drawing) -> Drawing:
        """
        Abstract method for colorizing a DXF drawing.

        Args:
            draw (Drawing): The DXF drawing to colorize.

        Returns
        -------
            Drawing: The colorized DXF drawing.
        """


class RendererStrategy(ABC):
    """Abstract base class for rendering a DXF drawing."""

    def __call__(self, draw: Drawing) -> Iterable[np.ndarray]:
        """
        Render the DXF drawing.

        Returns
        -------
        object:
        """
        return self._render(draw)

    @abstractmethod
    def _render(self, draw: Drawing) -> Iterable[np.ndarray]:
        """
        Abstract method for rendering a DXF drawing.

        Args:
            draw (Drawing): The DXF drawing to render.

        Returns
        -------
            np.ndarray: The rendered DXF drawing.
        """
