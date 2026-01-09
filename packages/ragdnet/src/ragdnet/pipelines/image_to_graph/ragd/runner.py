import time
import networkx as nx
import numpy as np

from ragdnet.pipelines.factory import ImageToGraphPipeline
from ragdnet.pipelines.image_to_graph.interfaces import (
    NUM_CHANNELS_GRAY,
    NUM_CHANNELS_RGB,
    GraphBuilderStrategy,
    LabelerStrategy,
    SegmentStrategy,
    WatershedStrategy,
)


class RagdnetPipeline(ImageToGraphPipeline):
    """Ragdnet pipeline for processing images into graphs."""

    def __init__(
        self,
        segmenter: SegmentStrategy,
        watershed: WatershedStrategy,
        graph_builder: GraphBuilderStrategy[np.ndarray],
        labeler: LabelerStrategy | None = None,
    ):
        self.segmenter = segmenter
        self.watershed = watershed
        self.graph_builder = graph_builder
        self.labeler = labeler

    def run(self, img: np.ndarray, *, verbose: bool = False) -> nx.Graph:
        vp = (lambda a: print(a)) if verbose else (lambda _: None)  # noqa: T201
        tic = time.perf_counter
        h, w = img.shape[:2]
        ch = img.shape[2] if img.ndim == NUM_CHANNELS_RGB else NUM_CHANNELS_GRAY
        vp(f"[IN] {w}x{h} ch={ch} {img.dtype}")

        t = tic()

        segmented = self.segmenter(img)
        vp(f"[1] seg {segmented.shape} {segmented.dtype} in {(tic() - t) * 1000:.1f}ms")
        t = tic()
        labels = self.watershed(segmented, img)
        vp(
            f"[2] ws labels={int(labels.max()) + 1 if labels.size else 0} in {
                (tic() - t) * 1000:.1f}ms"
        )

        t = tic()
        graph = self.graph_builder(labels)
        vp(
            f"[3] graph | V={graph.number_of_nodes()}"
            f"E={graph.number_of_edges()} in {(tic() - t) * 1000:.1f}ms"
        )

        if self.labeler is not None:
            t = tic()
            graph = self.labeler(
                graph,
                img if ch == NUM_CHANNELS_RGB else img,
            )
            vp(f"[4] label in {(tic() - t) * 1000:.1f}ms")

        return graph
