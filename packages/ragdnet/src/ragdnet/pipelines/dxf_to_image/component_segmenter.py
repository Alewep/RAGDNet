"""DXF component segmentation module."""

import copy

import networkx as nx
from ezdxf import bbox
from ezdxf.document import Drawing
from ezdxf.entities import DXFGraphic
from ezdxf.math import BoundingBox2d, Vec2

from ragdnet.pipelines.dxf_to_image.interfaces import SegmenterStrategy

TOLERANCE = 2


def is_starting_annotation(entity: DXFGraphic) -> bool:
    """Return True if the DXF entity is a DIMENSION annotation."""
    return entity.dxftype() == "DIMENSION"


def get_entity_bounding_boxes(entity: DXFGraphic) -> list[BoundingBox2d]:
    bb = bbox.extents([entity])
    verts = bb.rect_vertices()
    vec2s = [Vec2(p.x, p.y) for p in verts]
    return [BoundingBox2d(vec2s)]


def entities_connected(entity1: DXFGraphic, entity2: DXFGraphic) -> bool:
    """Return True if two DXF entities are connected via their bounding boxes."""
    for b1 in get_entity_bounding_boxes(entity1):
        for b2 in get_entity_bounding_boxes(entity2):
            if b1.has_overlap(b2):
                return True
    return False


class ComponentSegmenter(SegmenterStrategy):
    """Segment a DXF drawing into connected components."""

    def __init__(
        self,
        excluded_layers: list[str] | None = None,
        font_map: dict | None = None,
    ):
        """Initialize the segmenter with layers to exclude and a font mapping table."""
        if excluded_layers is None:
            self.excluded_layers = []
        else:
            self.excluded_layers = excluded_layers

        if font_map is None:
            self.font_map = {}
        else:
            self.font_map = font_map

    def replace_fonts(self, draw: Drawing) -> None:
        """Replace fonts in the drawing according to the mapping table."""
        for style in draw.styles:
            font_name = style.dxf.font
            if font_name in self.font_map:
                style.dxf.font = self.font_map[font_name]

    def _segment(self, doc: Drawing) -> list[Drawing]:
        """Segment the DXF document into sub-documents based on entity connectivity."""
        self.replace_fonts(doc)
        msp = doc.modelspace()
        items = [
            (entity, is_starting_annotation(entity))
            for entity in msp
            if entity.dxf.layer not in self.excluded_layers
        ]
        n = len(items)
        if n == 0:
            return []

        g: nx.Graph = nx.Graph()
        g.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                if entities_connected(items[i][0], items[j][0]):
                    g.add_edge(i, j)

        drawings = []
        for component in nx.connected_components(g):
            group = [items[i] for i in component]
            if any(is_ann for (_, is_ann) in group):
                new_doc = copy.deepcopy(doc)
                new_msp = new_doc.modelspace()
                for entity in list(new_msp):
                    new_msp.delete_entity(entity)
                for entity, _ in group:
                    new_msp.add_entity(entity.copy())
                drawings.append(new_doc)
        return drawings
