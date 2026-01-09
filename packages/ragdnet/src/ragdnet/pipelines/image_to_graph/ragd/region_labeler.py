"""Labeler that assigns a label to each node based on its source pixel color."""

import networkx as nx
import numpy as np

from ragdnet.utils.color_mapper import map_color_to_id_image

from ragdnet.pipelines.image_to_graph.interfaces import (
    LabelerStrategy,
)


class RegionLabeler(LabelerStrategy):
    def _label(self, graph: nx.Graph, colored_img: np.ndarray) -> nx.Graph:
        """
        Assigns to each node a label based on the majority color
        of the pixels belonging to its region in the colored image.

        The region is given by `element.source_pixels`, an array of (row, col)
        pixel coordinates belonging to the region.
        """
        height, width, _ = colored_img.shape
        # ids_img: integer ID for each pixel color
        ids_img = map_color_to_id_image(colored_img)
        for node, attr in graph.nodes(data=True):
            coords = attr["src"]
            if coords is None:
                continue

            coords = np.asarray(coords, dtype=int)
            if coords.size == 0:
                continue

            # Expect shape (N, 2): (row, col)
            if coords.ndim != 2 or coords.shape[1] != 2:
                raise ValueError(
                    f"source_pixels must have shape (N, 2), got {coords.shape}"
                )

            rows, cols = coords[:, 0], coords[:, 1]

            # Check that all coordinates are within image bounds
            if (
                (rows < 0).any()
                or (rows >= height).any()
                or (cols < 0).any()
                or (cols >= width).any()
            ):
                raise ValueError(
                    "Some source_pixels coordinates are out of image bounds: "
                    f"image={(height, width)}, rows∈[{rows.min()}, {rows.max()}], "
                    f"cols∈[{cols.min()}, {cols.max()}]"
                )

            # Get color IDs for all pixels in the region
            region_ids = ids_img[rows, cols].ravel()
            if region_ids.size == 0:
                continue

            # Majority color ID
            vals, counts = np.unique(region_ids, return_counts=True)
            major_id = vals[np.argmax(counts)]

            graph.nodes[node]["y"] = int(major_id)

        return graph
