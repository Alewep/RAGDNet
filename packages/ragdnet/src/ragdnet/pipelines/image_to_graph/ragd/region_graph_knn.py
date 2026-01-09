import math
import cv2
import networkx as nx
import numpy as np
from shapely.geometry import Polygon

from ragdnet.pipelines.image_to_graph.interfaces import GraphBuilderStrategy
from shapely.ops import nearest_points

EPSILON_POLY = math.sqrt(2)
MIN_POLY_SIZE = 3


def get_polygon(mask: np.ndarray) -> Polygon:
    mask = mask.astype(np.uint8) * 255
    mask = np.ascontiguousarray(mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        raise ValueError("No contour found")
    if len(contours) > 1:
        raise ValueError("More than one contour found")

    poly = cv2.approxPolyDP(contours[0], EPSILON_POLY, closed=True)
    poly = poly.squeeze()

    if poly.ndim == 1:
        poly = poly.reshape(1, 2)

    if poly.shape[0] < MIN_POLY_SIZE:
        x, y, w, h = cv2.boundingRect(contours[0])
        rect = [
            (x - 0.5, y - 0.5),
            (x + w - 0.5, y - 0.5),
            (x + w - 0.5, y + h - 0.5),
            (x - 0.5, y + h - 0.5),
        ]
        return Polygon(rect)

    return Polygon(poly)


def get_src_coords(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return np.empty((0, 2), dtype=int)
    # (y, x) pairs
    return np.stack([ys.astype(int), xs.astype(int)], axis=1)


class RegionGraphKNN(GraphBuilderStrategy[np.ndarray]):
    """Build a directed KNN graph from labeled regions."""

    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k

    def _build_graph(self, label_img: np.ndarray) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.graph["img_shape"] = label_img.shape

        labels = np.unique(label_img)
        labels = labels[labels != 0]
        labels = np.sort(labels)

        polygons: dict[int, Polygon] = {}
        for lab_idx, actual_label in enumerate(labels):
            mask = label_img == actual_label
            poly = get_polygon(mask)
            polygons[lab_idx] = poly
            poly_coords = list(poly.exterior.coords)

            centroid = poly.centroid
            graph.add_node(
                int(lab_idx),
                x=poly_coords,
                src=get_src_coords(mask),
                centroid=(float(centroid.x), float(centroid.y)),
                label=int(actual_label),
            )

        node_ids, neighbors_ids, dist_knn = RegionGraphKNN.topological_knn(
            polygons=polygons,
            k=self.k,
            normalize=True,
        )

        # Directed edges: u -> its k nearest neighbors
        for i in range(node_ids.shape[0]):
            u = int(node_ids[i])
            for j in range(neighbors_ids.shape[1]):
                v = int(neighbors_ids[i, j])
                d = float(dist_knn[i, j])

                if u == v:
                    continue

                Pu = polygons[u]
                Pv = polygons[v]
                p_u, p_v = nearest_points(Pu, Pv)
                dx = float(p_v.x - p_u.x)
                dy = float(p_v.y - p_u.y)
                norm = math.hypot(dx, dy)
                ux, uy = (dx / norm, dy / norm) if norm > 0 else (0.0, 0.0)

                graph.add_edge(v, u, edge_attr=[d, ux, uy])

        return graph

    @staticmethod
    def topological_knn(
        polygons: dict[int, Polygon],
        k: int,
        normalize: bool = True,
    ) -> tuple[
        np.ndarray,  # node_ids, shape (n,)
        np.ndarray,  # neighbors_ids, shape (n, k)
        np.ndarray,  # dist_knn, shape (n, k)
    ]:
        """
        Compute k nearest neighbors in polygon space (topological distance).

        If normalize is True, distances on KNN edges are divided by the
        global maximum distance among all KNN edges in the graph.
        """
        node_ids = np.array(sorted(polygons.keys()), dtype=int)
        n = node_ids.shape[0]

        if n == 0 or k <= 0:
            return (
                node_ids,
                np.empty((0, 0), dtype=int),
                np.empty((0, 0), dtype=float),
            )

        k = min(k, n - 1)
        if k <= 0:
            # Single node case
            return (
                node_ids,
                np.empty((n, 0), dtype=int),
                np.empty((n, 0), dtype=float),
            )

        topo_dist = np.full((n, n), np.inf, dtype=float)

        for i in range(n):
            pi = polygons[int(node_ids[i])]
            for j in range(i + 1, n):
                pj = polygons[int(node_ids[j])]
                d = float(pi.distance(pj))
                topo_dist[i, j] = topo_dist[j, i] = d

        # Exclude self from KNN search
        np.fill_diagonal(topo_dist, np.inf)

        row_idx = np.arange(n)[:, None]  # (n, 1)
        knn_idx = np.argpartition(topo_dist, k, axis=1)[:, :k]  # (n, k)
        dist_knn = topo_dist[row_idx, knn_idx]  # (n, k)

        order = np.argsort(dist_knn, axis=1)
        knn_idx = knn_idx[row_idx, order]
        dist_knn = dist_knn[row_idx, order]

        if normalize:
            max_d = np.max(dist_knn)
            if np.isfinite(max_d) and max_d > 0.0:
                dist_knn = dist_knn / max_d
            else:
                dist_knn.fill(0.0)

        neighbors_ids = node_ids[knn_idx]

        return node_ids, neighbors_ids, dist_knn
