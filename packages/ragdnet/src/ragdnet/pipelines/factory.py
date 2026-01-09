"""Pipeline factories and dynamic strategy loading from TOML configurations."""

from __future__ import annotations
import importlib
import inspect
import json
from pathlib import Path
import pkgutil
import tomllib
from abc import ABC, abstractmethod
from collections.abc import Iterator
from types import ModuleType
from typing import Generic, TypeVar
import cv2
import ezdxf
from ezdxf.document import Drawing
import ezdxf.recover
from networkx.readwrite import json_graph
import numpy as np
from ragdnet.pipelines.image_to_graph.interfaces import nx

InT = TypeVar("InT")
OutT = TypeVar("OutT")

# Type aliases for clarity
ImageArray = np.ndarray

class Pipeline(ABC, Generic[InT, OutT]):
    """
    Generic pipeline.

    - run(data): works on in-memory data (no disk I/O)
    - run_io(input_path, output_path): load from disk, call run, save to disk
    """

    # ==== Pure API, no I/O ====
    @abstractmethod
    def run(self, data: InT) -> OutT:
        """
        Process in-memory data and return the result.
        No disk I/O here.
        """

    # ==== I/O layer, built on top of run ====

    @abstractmethod
    def _load(self, path: Path) -> InT:
        """Load raw data from a file path."""

    @abstractmethod
    def _save(self, result: OutT, path: Path, name:str) -> None:
        """Save the result to a file path."""

    def run_io(self, input_path: Path | str, output_path: Path | str) -> None:
        """
        If input_path is a file: process ONE file -> ONE file.
        If input_path is a directory: process all files in the directory -> output directory.

        output_path:
        - if input_path is a file: output_path must be a file path
        - if input_path is a directory: output_path is a directory (created if needed)
        """
        in_p = Path(input_path)
        out_p = Path(output_path)

        if in_p.is_file():
            # Case 1: single file
            out_p.parent.mkdir(parents=True, exist_ok=True)
            data = self._load(in_p)
            result = self.run(data)
            self._save(result, out_p, in_p.name.split('.')[0])
        else:
            raise FileNotFoundError(f"Input path does not exist: {in_p}")


class DxfToImagePipeline(Pipeline[Drawing, ImageArray], ABC):
    """
    Abstract DXF -> image pipeline.

    - _load: read DXF file with ezdxf
    - _save: write image with cv2.imwrite
    - run: implement DXF document -> image (numpy array) in subclasses
    """

    def _load(self, path: Path) -> Drawing:
        """Load a DXF document from disk using ezdxf."""
        return ezdxf.recover.readfile(str(path))[0]

    def _save(self, result: ImageArray, path: Path, name:str) -> None:
        """Save an image (numpy array) to disk using cv2."""
       
        for i,img in enumerate(result):
            current_path = path / f"{name}_{i}.png"

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if not cv2.imwrite(str(current_path), img):
                raise IOError(f"Failed to write image to {path}")         

    @abstractmethod
    def run(self, data: Drawing) -> ImageArray:
        """
        Convert a DXF document to an image (numpy array, BGR for OpenCV).

        You implement the rendering logic here in a concrete subclass.
        """
        raise NotImplementedError


class ImageToGraphPipeline(Pipeline[ImageArray, nx.Graph], ABC):
    """
    Abstract image -> NetworkX graph pipeline.

    - _load: read image with cv2.imread
    - run: implement image -> nx.Graph in subclasses
    - _save: save the graph as node-link JSON
    """

    def _load(self, path: Path) -> ImageArray:
        """Load an image from disk using cv2."""
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        return img

    @abstractmethod
    def run(self, data: ImageArray) -> nx.Graph:
        """
        Convert an image (numpy array) to a NetworkX graph.

        You define the graph structure (nodes/edges/attributes) in your subclass.
        For example, you can tag nodes with attributes like:
            node["kind"] in {"dimension", "symbol", "text", ...}
        """
        raise NotImplementedError

    def _save(self, result: nx.Graph, path: Path, name:str) -> None:
        path = path / (name + ".json")
        data = json_graph.node_link_data(result)

        def default(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, np.generic):
                return o.item()
            if isinstance(o, Path):
                return str(o)
            if isinstance(o, set):
                return list(o)
            raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

        json_text = json.dumps(data, ensure_ascii=False, indent=2, default=default)
        path.write_text(json_text, encoding="utf-8")


def load_strategy(class_name: str, params: dict, package: str) -> object:
    """
    Find and instantiate a class by name inside a package tree.

    - class_name: target class name
    - params: initialization parameters
    - package: root package path (e.g., "ragdnet.pipelines.dxf_to_image")
    """
    # 1) Canonical import of the root package
    try:
        root_pkg: ModuleType = importlib.import_module(package)
    except ImportError as err:
        raise ImportError(f"Cannot import package '{package}'") from err

    # 2) Enumerate submodules using a canonical prefix
    def walk_modules(pkg: ModuleType) -> Iterator[str]:
        yield pkg.__name__
        if hasattr(pkg, "__path__"):  # only for packages
            for _, name, _ in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
                yield name

    # 3) Scan modules and look for the class
    for mod_name in walk_modules(root_pkg):
        try:
            module = importlib.import_module(mod_name)

        except Exception:
            continue

        cls = getattr(module, class_name, None)
        if inspect.isclass(cls) and getattr(cls, "__module__", None) == module.__name__:
            return cls(**(params or {}))

    raise AttributeError(
        f"Class '{class_name}' not found in '{package}' or its subpackages"
    )


def create_pipeline(
    config_path: str, pipeline_class: type[Pipeline], package: str
) -> Pipeline:
    """
    Load a pipeline configuration from a TOML file and build the pipeline.

    The TOML should have top-level entries where each key is a strategy name
    and its value is a table with 'class' and optionally 'params'.

    Example config.toml:
    ```toml
    segmenter = { class = "ComponentSegmenter", params = {} }
    colorizer = { class = "CategoricalColorizer", params = {} }
    renderer = { class = "PlotRenderer", params = {dpi=300} }
    ```
    :param config_path: path to the TOML config file
    :param pipeline_class: pipeline class to instantiate
    :param package: package where strategy classes reside
    :returns: instance of pipeline_class initialized with the strategies
    """
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    strategies = {}
    # Each top-level key is a strategy
    for name, spec in cfg.items():
        cls_name = spec.get("class", None)  # class name to load
        if cls_name is not None:
            params = spec.get("params", {})  # optional init params
            strategies[name] = load_strategy(cls_name, params, package)

    # Instantiate pipeline with strategies as kwargs
    return pipeline_class(**strategies)
