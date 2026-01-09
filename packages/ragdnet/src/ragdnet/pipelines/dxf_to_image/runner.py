"""Pipeline for converting DXF files.

This module provides a pipeline to process DXF files into images using
segmentation, colorization, and rendering strategies.
"""

from collections.abc import Iterable
import numpy as np
from ezdxf.document import Drawing

from ragdnet.pipelines.dxf_to_image.interfaces import (
    ColorizerStrategy,
    RendererStrategy,
    SegmenterStrategy,
)
from ragdnet.pipelines.factory import DxfToImagePipeline


class DXFImagePipeline(DxfToImagePipeline):
    """Convert a DXF into images.

    Process a DXF file with segmentation, colorisation and rendering
    strategies to produce a final images.
    """

    def __init__(
        self,
        colorizer: ColorizerStrategy,
        renderer: RendererStrategy,
        segmenter: SegmenterStrategy | None = None,
    ):
        """Initialize the DXFImagePipeline.

        Args:
            segmenter (SegmenterStrategy | None): The segmentation strategy to use.
            colorizer (ColorizerStrategy): The colorization strategy to use.
            renderer (RendererStrategy): The rendering strategy to use.
        """
        self.colorizer = colorizer
        self.segmenter = segmenter
        self.renderer = renderer

    def run(self, draw: Drawing) -> Iterable[np.ndarray]:
        """Process a DXF drawing into images.

        Args:
            draw (Drawing): The DXF drawing to process.

        Returns
        -------
            Iterable[Iterable[np.ndarray]]: A nested iterable of image arrays.
        """
        drawings: Iterable[Drawing] = self.segmenter(draw) if self.segmenter else [draw]
        imgs = []
        for drawing in drawings:
            colored = self.colorizer(drawing)
            render = self.renderer(colored)
            imgs.extend(render)
        return imgs

