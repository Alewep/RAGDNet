"""Module for rendering DXF drawings to RGB images using Matplotlib."""

from collections.abc import Iterable

import cv2
import ezdxf.addons.drawing.matplotlib as ezdxf_mpl
import matplotlib as mpl
import numpy as np
from ezdxf.addons.drawing import Frontend, RenderContext, config
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
from ezdxf.document import Drawing
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from ragdnet.pipelines.dxf_to_image.interfaces import RendererStrategy


class PlotRenderer(RendererStrategy):
    """Rendering strategy using Matplotlib to convert a DXF drawing to an RGB image."""

    def __init__(self, dpi: int) -> None:
        self.dpi = dpi

    def _render(self, draw: Drawing) -> Iterable[np.ndarray]:
        """
        Convert a DXF drawing to an RGB image using Matplotlib.

        This version uses a transparent background to better preserve DXF colors.
        """
        ezdxf_mpl.SCATTER_POINT_SIZE = 0
        mpl.rcParams["lines.antialiased"] = False
        mpl.rcParams["patch.antialiased"] = False
        mpl.rcParams["text.antialiased"] = False
        mpl.rcParams["path.simplify"] = False
        mpl.rcParams["path.simplify_threshold"] = 1.0
        mpl.use("Agg")

        plt.close("all")
        fig, ax = plt.subplots(dpi=self.dpi)
        ctx = RenderContext(draw)
        backend = MatplotlibBackend(ax)
        # Use the default configuration so no color override is applied:
        cfg = config.Configuration(background_policy=config.BackgroundPolicy.WHITE)
        frontend = Frontend(ctx, backend, config=cfg)
        frontend.draw_layout(draw.modelspace(), finalize=True)
        ax.set_aspect("equal")

        fig.canvas.draw()
        if isinstance(fig.canvas, FigureCanvasAgg):
            height, width = fig.canvas.get_width_height()[::-1]
            rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            image_rbga: np.ndarray = rgba.reshape((height, width, 4))
            image_rbg: np.ndarray = cv2.cvtColor(image_rbga, cv2.COLOR_RGBA2RGB)
        else:
            raise TypeError("Unsupported backend")

        plt.close(fig)
        return [image_rbg]
