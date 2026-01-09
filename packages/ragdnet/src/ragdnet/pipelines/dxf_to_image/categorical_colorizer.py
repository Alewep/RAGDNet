"""Module for categorical coloring of DXF entities."""

import copy
import logging
from contextlib import suppress

from ezdxf.disassemble import recursive_decompose
from ezdxf.document import Drawing
from ezdxf.entities import DXFGraphic
from ezdxf.layouts import Modelspace

from ragdnet.pipelines.dxf_to_image.interfaces import ColorizerStrategy
from ragdnet.utils.color_mapper import map_id_to_color

logger = logging.getLogger(__name__)


# ColorMapping now stores integer IDs (not RGB tuples)
ColorMapping = dict[str, dict[str, int]]


class CategoricalColorizer(ColorizerStrategy):
    """Categorical colorizer for DXF entities based on layers and types."""

    def __init__(self, settings: dict):
        """Initialize the colorizer with the given settings.

        Args:
            settings: Dictionary containing coloring parameters
        """
        self.settings = settings

    def _create_color_mapping(self) -> ColorMapping:
        """Create the color mapping for layers and types.

        Returns
        -------
        Dictionary containing the color mappings
        """
        # Store integer color IDs; RGB is computed later
        colors: ColorMapping = {"layer": {}, "type": {}}
        color = 1
        for layer in self.settings["layer"]:
            colors["layer"][layer] = color
            color += 1
        for layer in self.settings["type"]:
            colors["type"][layer] = color
            color += 1
        return colors

    def _process_entity(
        self, entity: DXFGraphic, colors: ColorMapping, new_msp: Modelspace
    ) -> list[DXFGraphic]:
        """Process an entity and its sub-entities for coloring.

        Args:
            entity: The DXF entity to process
            colors: The color mapping
            new_msp: The destination modelspace
        """
        copies: list[DXFGraphic] = []
        try:
            sub_entities = list(recursive_decompose([entity]))
        except (ValueError, TypeError, AttributeError) as e:
            logger.warning(f"Error while decomposing entity: {e}")
            return copies

        for sub_entity in sub_entities:
            if not isinstance(sub_entity, DXFGraphic):
                continue
            self._apply_color_to_entity(entity, sub_entity, colors)
            with suppress(Exception):
                c = sub_entity.copy()
                new_msp.add_entity(c)
                copies.append(c)
        return copies

    def _apply_color_to_entity(
        self, entity: DXFGraphic, sub_entity: DXFGraphic, colors: ColorMapping
    ) -> None:
        """Apply the appropriate color to a sub-entity.

        Args:
            entity: The parent entity
            sub_entity: The sub-entity to color
            colors: The color mapping
        """
        # Compute RGB from stored integer ID at apply-time
        color_id = None
        if entity.dxf.layer in self.settings["layer"]:
            color_id = colors["layer"][entity.dxf.layer]

        if sub_entity.dxf.dxftype in self.settings["type"]:
            color_id = colors["type"][sub_entity.dxf.dxftype]

        if color_id is not None:
            sub_entity.rgb = map_id_to_color(color_id)
        else:
            # Pure geometry case not considered as annotation
            sub_entity.rgb = (0, 0, 0)

    # Small helpers to keep _colorize readable
    def _priority(self, entity: DXFGraphic, colors: ColorMapping) -> int:
        # lower id => should be drawn on top (last)
        layer = getattr(entity.dxf, "layer", None)
        dxftype = getattr(entity.dxf, "dxftype", None)
        if layer in colors["layer"]:
            return colors["layer"][layer]
        if dxftype in colors["type"]:
            return colors["type"][dxftype]
        return -1

    def _colorize(self, doc: Drawing) -> Drawing:
        """
        Colorize the entities of the DXF document according to the defined categories.

        Args:
            doc: The DXF document to colorize

        Returns
        -------
        The colorized DXF document
        """
        msp = doc.modelspace()
        new_doc = copy.deepcopy(doc)
        new_msp = new_doc.modelspace()

        # Remove all existing entities
        for entity in list(new_msp):
            new_msp.delete_entity(entity)

        colors = self._create_color_mapping()

        # Sort so smaller IDs are last (drawn on top by redraw order)
        entities_sorted = sorted(
            msp, key=lambda e: self._priority(e, colors), reverse=True
        )

        # Copy + color, and keep exact draw order locally
        drawn: list[DXFGraphic] = []
        for entity in entities_sorted:
            drawn.extend(self._process_entity(entity, colors, new_msp))

        # Enforce redraw order (MatplotlibBackend uses this, not insertion order)
        handles = [
            e.dxf.handle for e in drawn if hasattr(e.dxf, "handle") and e.dxf.handle
        ]
        with suppress(Exception):
            if handles:
                new_msp.set_redraw_order(handles)

        return new_doc
