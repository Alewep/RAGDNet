
import numpy as np
import copy
import os
import concurrent.futures

import numpy as np
import cv2
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import from_networkx
import networkx as nx
from tqdm import tqdm

from ragdnet.learning.datasets.base import AlignedImageDatasetBase
from ragdnet.pipelines.image_to_graph.ragd.runner import RagdnetPipeline
from ragdnet.pipelines.factory import create_pipeline
from ragdnet.utils.color_mapper import map_color_to_id_image




class PointData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "point2node":
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ("points", "point2node"):
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)


# -----------------------------
# Validation
# -----------------------------

def assert_valid_pyg_graph(
    data: PointData,
    num_classes: int | None = None,
    expect_points: bool = True,
) -> None:
    """Minimal sanity checks for a PyG graph."""

    # Basic graph structure
    assert isinstance(data, Data), f"Expected Data, got {type(data)}"
    assert hasattr(data, "num_nodes") and isinstance(data.num_nodes, int)
    assert data.num_nodes >= 0

    # Edge index
    assert hasattr(data, "edge_index"), "edge_index missing"
    ei = data.edge_index
    assert isinstance(ei, torch.Tensor)
    assert ei.ndim == 2 and ei.size(0) == 2, (
        f"edge_index must be [2, E], got {tuple(ei.size())}"
    )
    if ei.numel() > 0:
        assert int(ei.min()) >= 0, "edge_index has negative indices"
        assert int(ei.max()) < data.num_nodes, "edge_index has indices >= num_nodes"

    # Points / point2node
    if expect_points:
        assert hasattr(data, "points"), "points missing"
        assert hasattr(data, "point2node"), "point2node missing"
        pts = data.points
        p2n = data.point2node

        assert isinstance(pts, torch.Tensor) and pts.ndim == 2 and pts.size(1) == 2
        assert pts.dtype == torch.float32
        assert isinstance(p2n, torch.Tensor) and p2n.ndim == 1
        assert p2n.size(0) == pts.size(0)
        assert p2n.dtype == torch.long
        if p2n.numel() > 0:
            assert int(p2n.min()) >= 0, "point2node has negative indices"
            assert int(p2n.max()) < data.num_nodes, (
                "point2node has indices >= num_nodes"
            )

    # Labels in [0, num_classes - 1]
    if num_classes is not None and hasattr(data, "y"):
        y = data.y.view(-1)
        assert isinstance(y, torch.Tensor) and y.numel() > 0
        assert y.dtype in (torch.long, torch.int64, torch.int32, torch.int16)
        assert int(y.min()) >= 0, "y contains negative labels"
        assert int(y.max()) < num_classes, "y contains labels >= num_classes"


# -----------------------------
# Data -> PointData (build points + point2node)
# -----------------------------

def data_to_pointdata(
    data: Data,
    attr_name: str = "x",
) -> PointData:
    """
    Convert data.<attr_name> (list of 2D point clouds per node) into:
      - points     : [M, 2] float32
      - point2node : [M]   int64
    Then deletes <attr_name> and returns a PointData instance.

    Uses the constructor pattern:
        pd = PointData()
        for k, v in data.items():
            pd[k] = v
    """

    if not hasattr(data, attr_name):
        raise AttributeError(
            f"{attr_name} does not exist in data (attrs: {list(data.keys())})"
        )

    node_clouds = getattr(data, attr_name)
    points_list = []
    point2node_list = []

    for node_idx, pts in enumerate(node_clouds):
        pts = torch.as_tensor(pts, dtype=torch.float32)

        if pts.ndim == 1:
            pts = pts.view(-1, 2)

        if pts.size(-1) != 2:
            raise ValueError(f"Expected 2D points, got shape={pts.shape}")

        points_list.append(pts)
        point2node_list.append(
            torch.full((pts.size(0),), int(node_idx), dtype=torch.long)
        )

    # Safe concat if empty
    data.points = (
        torch.cat(points_list, dim=0)
        if points_list
        else torch.empty((0, 2), dtype=torch.float32)
    )
    data.point2node = (
        torch.cat(point2node_list, dim=0)
        if point2node_list
        else torch.empty((0,), dtype=torch.long)
    )

    delattr(data, attr_name)

    # Build PointData with the requested pattern
    pd = PointData()
    for k, v in data.items():
        pd[k] = v

    # Ensure num_nodes is preserved
    if hasattr(data, "num_nodes"):
        pd.num_nodes = data.num_nodes

    return pd


# -----------------------------
# NetworkX -> PointData
# -----------------------------

def nx_to_pointdata(
    graph_nx: nx.Graph,
    keep_only=("x", "y", "src"),
) -> PointData:
    """
    Minimal conversion for a drawing-graph model.
    Only node attributes listed in `keep_only` are preserved.
    """

    graph_nx = copy.deepcopy(graph_nx)
    keep_only = set(keep_only)
    src_list = []

    for n in graph_nx.nodes:
        node = graph_nx.nodes[n]
        src_list.append(node.get("src"))

        for k in list(node.keys()):
            if k not in keep_only:
                node.pop(k, None)

        node.pop("src", None)

    data = from_networkx(graph_nx)
    data.num_nodes = graph_nx.number_of_nodes()

    # Convert node attribute `x` -> points/point2node and return PointData
    pd = data_to_pointdata(data, attr_name="x")

    # Add src after conversion (kept as non-tensor object array)
    pd.src = np.array(src_list, dtype=object)

    return pd


# -----------------------------
# Worker
# -----------------------------

def build_graph_worker(args: tuple[str, str]) -> PointData:
    """
    Worker function used by ProcessPoolExecutor.
    Creates its own pipeline instance, builds a graph for one image,
    converts to PointData and returns it.
    """
    img_path, config_path = args
    pipeline = create_pipeline(config_path, RagdnetPipeline, "ragdnet.pipelines")

    img = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    graph_nx = pipeline.run(img)

    if graph_nx is None:
        raise RuntimeError(f"Graph build returned None for image: {img_path}")

    if not isinstance(graph_nx, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError(
            f"Unexpected graph type for image: {img_path} "
            f"(got {type(graph_nx)})"
        )

    graph_py = nx_to_pointdata(graph_nx)
    graph_py.img_path = img_path

    return graph_py


# -----------------------------
# Dataset
# -----------------------------

class GraphDrawingDataset(AlignedImageDatasetBase, Dataset):
    def __init__(
        self,
        root: str,
        config_path: str,
        img_transform=None,
        num_workers=None,
        train: bool = True,
        num_classes: int | None = None,
    ):
        self.config_path = config_path
        self.img_transform = img_transform
        self.num_workers = num_workers
        self._num_classes = num_classes

        raw_root = os.path.join(root, "raw")
        AlignedImageDatasetBase.__init__(self, root=raw_root, train=train)

        Dataset.__init__(
            self,
            root=root,
            transform=None,
            pre_transform=None,
        )

    @property
    def raw_file_names(self):
        return [os.path.relpath(p, self.raw_dir) for p in self.image_paths] or ["dummy"]

    @property
    def processed_file_names(self):
        cfg_base = os.path.basename(self.config_path)
        cfg_stem = os.path.splitext(cfg_base)[0]

        subdir = cfg_stem

        if len(self.image_paths) == 0:
            return [os.path.join(subdir, f"empty_ragdnet.pt")]

        return [
            os.path.join(subdir, f"data_ragdnet_{i}.pt")
            for i in range(len(self.image_paths))
        ]

    def len(self):
        return len(self.image_paths)

    def _save_graph(self, out_path: str, graph_py: PointData, pbar: tqdm):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        assert_valid_pyg_graph(
            graph_py,
            num_classes=self._num_classes,
            expect_points=True,
        )

        torch.save(graph_py, out_path)
        pbar.update(1)

    def process(self):
        img_paths = self.image_paths
        num_imgs = len(img_paths)

        if num_imgs == 0:
            out_path = self.processed_paths[0]
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            torch.save(None, out_path)
            return

        first_out_dir = os.path.dirname(self.processed_paths[0])
        os.makedirs(first_out_dir, exist_ok=True)

        indices_to_process = [
            i
            for i, out_path in enumerate(self.processed_paths)
            if not os.path.exists(out_path)
        ]
        paths_to_process = [img_paths[i] for i in indices_to_process]

        use_parallel = self.num_workers and self.num_workers > 1
        with tqdm(total=num_imgs, desc="Building graphs", unit="image") as pbar:
            num_skipped = num_imgs - len(indices_to_process)
            if num_skipped > 0:
                pbar.update(num_skipped)

            if use_parallel:
                tasks = [
                    (img_path, self.config_path)
                    for img_path in paths_to_process
                ]

                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=self.num_workers
                ) as executor:
                    for local_idx, graph_py in enumerate(
                        executor.map(build_graph_worker, tasks)
                    ):
                        i = indices_to_process[local_idx]
                        out_path = self.processed_paths[i]
                        self._save_graph(out_path, graph_py, pbar)
            else:
                for local_idx, img_path in enumerate(paths_to_process):
                    i = indices_to_process[local_idx]
                    out_path = self.processed_paths[i]

                    graph_py = build_graph_worker(
                        (img_path, self.pipeline_name, self.config_path)
                    )
                    self._save_graph(out_path, graph_py, pbar)

    def get(self, idx):
        data = torch.load(self.processed_paths[idx], weights_only=False)

        if data is None:
            raise RuntimeError(f"Processed graph is None for idx={idx}")

        if not isinstance(data, PointData):
            raise TypeError(
                f"Expected PointData at idx={idx}, got {type(data)}. "
                "Your processing step must save PointData objects."
            )

        if hasattr(data, "img_path"):
            img_rgb = cv2.imread(data.img_path, cv2.IMREAD_COLOR_RGB)
            if img_rgb is None:
                raise RuntimeError(f"Failed to read image: {data.img_path}")

            img_mask = map_color_to_id_image(img_rgb)
            data.img_mask = img_mask

        return data
