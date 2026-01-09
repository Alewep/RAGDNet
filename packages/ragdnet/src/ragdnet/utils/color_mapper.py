"""Utility functions for 3-byte (RGB) color permutation and conversion."""

import numpy as np

BLANK_VALUE = 256 * 256 * 256 - 1  # 0xFFFFFF = white (blank color)
BLANK_ID = -1  # blank ID
SEED = 5079232


def map_id_to_color(color_id: int, seed: int = SEED) -> tuple[int, int, int]:
    """Map an integer ID to a pseudo-random RGB color in a bijective way.

    Notes
    -----
    - BLANK_ID (-1) is always mapped to white (0xFFFFFF).
    - All other IDs (0..0xFFFFFE) are bijectively permuted within [0, 0xFFFFFE].
    """
    if not (color_id == BLANK_ID or 0 <= color_id < BLANK_VALUE):
        raise ValueError(
            f"'color_id' must be BLANK_ID ({BLANK_ID}) or in the range [0, {BLANK_VALUE - 1}]. "
            f"Received: {color_id}"
        )

    if color_id == BLANK_ID:
        permuted = BLANK_VALUE  # fixed to white
    else:
        # Bijective permutation over [0, BLANK_VALUE - 1]
        permuted = (color_id * seed) % BLANK_VALUE

    red = (permuted >> 16) & 0xFF
    green = (permuted >> 8) & 0xFF
    blue = permuted & 0xFF
    return red, green, blue


def map_color_to_id(red: int, green: int, blue: int, seed: int = SEED) -> int:
    """Inverse mapping: from RGB color back to the original integer ID.

    Notes
    -----
    - White (0xFFFFFF) is always mapped to BLANK_ID (-1).
    """
    for name, c in (("red", red), ("green", green), ("blue", blue)):
        if not (0 <= c <= 255):
            raise ValueError(f"'{name}' must be in [0, 255]. Received: {c}")

    packed = (red << 16) | (green << 8) | blue

    if packed == BLANK_VALUE:
        # White → BLANK_ID
        return BLANK_ID

    seed_inverse = pow(seed, -1, BLANK_VALUE)
    return (packed * seed_inverse) % BLANK_VALUE


def map_id_to_color_image(ids, seed: int = SEED) -> np.ndarray:
    """Map one or more integer IDs to RGB color values.

    Notes
    -----
    - IDs can be BLANK_ID (-1) or within [0, BLANK_VALUE - 1].
    - BLANK_ID is mapped to white (0xFFFFFF).
    """
    ids_arr = np.asarray(ids, dtype=np.int64)

    mask_blank = ids_arr == BLANK_ID

    # Validate input values
    invalid_neg = (ids_arr < 0) & (~mask_blank)
    invalid_high = (ids_arr >= BLANK_VALUE) & (~mask_blank)
    if invalid_neg.any() or invalid_high.any():
        raise ValueError(
            f"'ids' must be BLANK_ID ({BLANK_ID}) or in [0, {BLANK_VALUE - 1}]. "
            f"Received min={ids_arr.min()}, max={ids_arr.max()}."
        )

    # Bijective permutation on [0, BLANK_VALUE - 1]
    permuted = (ids_arr * seed) % BLANK_VALUE

    # BLANK_IDs are always mapped to white
    permuted = np.where(mask_blank, BLANK_VALUE, permuted)

    # Extract RGB channels
    red = ((permuted >> 16) & 0xFF).astype(np.uint8)
    green = ((permuted >> 8) & 0xFF).astype(np.uint8)
    blue = (permuted & 0xFF).astype(np.uint8)

    image = np.stack([red, green, blue], axis=-1)
    return image


def map_color_to_id_image(image: np.ndarray, seed: int = SEED) -> np.ndarray:
    """Inverse mapping: converts an RGB image back to integer IDs.

    Notes
    -----
    - White pixels (0xFFFFFF) are mapped back to BLANK_ID (-1).
    """
    if image.shape[-1] != 3:
        raise ValueError(
            f"'image' must have shape (..., 3) for RGB. Got shape: {image.shape}."
        )

    img = np.asarray(image, dtype=np.int64)

    r = img[..., 0]
    g = img[..., 1]
    b = img[..., 2]

    if ((r < 0) | (r > 255) | (g < 0) | (g > 255) | (b < 0) | (b > 255)).any():
        raise ValueError("All RGB components must be in [0, 255].")

    # Pack RGB components into a single 24-bit integer
    packed = (r << 16) | (g << 8) | b

    # Apply the inverse permutation (on [0, BLANK_VALUE - 1])
    seed_inverse = pow(seed, -1, BLANK_VALUE)
    ids = (packed * seed_inverse) % BLANK_VALUE

    # White → BLANK_ID
    mask_white = packed == BLANK_VALUE
    ids = np.where(mask_white, BLANK_ID, ids)

    return ids.astype(np.int64)
