import math
from typing import Tuple
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label
from skimage.morphology import skeletonize

from ragdnet.pipelines.image_to_graph.interfaces import KERNEL, WHITE, SegmentStrategy

MIN_JUNCTION_DEGREE = 3



def order_branch(branch_mask: np.ndarray) -> np.ndarray:
    """
    Order the coordinates of a branch (binary or boolean mask).

    Parameters
    ----------
    branch_mask : np.ndarray
        Binary mask of the branch.

    Returns
    -------
    np.ndarray
        Ordered coordinates of the branch pixels.
    """
    coords = np.column_stack(np.where(branch_mask))
    if len(coords) == 0:
        return np.empty((0, 2), dtype=int)

    neighbors = ndi.convolve(branch_mask.astype(int), KERNEL, mode="constant")
    endpoints = [tuple(c) for c in coords if neighbors[tuple(c)] == 1]

    cycle = False
    if len(endpoints) == 0:
        start = tuple(coords[0])
        cycle = True
    else:
        start = endpoints[0]

    current = start
    visited = {current}
    ordered = [current]
    has_next = True

    while has_next:
        has_next = False
        r, c = current
        directions = [
            (dr, dc) for dr in [-1, 0, 1] for dc in [-1, 0, 1] if (dr, dc) != (0, 0)
        ]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (nr, nc) in visited:
                continue

            if (
                0 <= nr < branch_mask.shape[0]
                and 0 <= nc < branch_mask.shape[1]
                and branch_mask[nr, nc]
            ):
                current = (nr, nc)
                visited.add(current)
                ordered.append(current)
                has_next = True
                break

    # Pour les cycles, on ferme le chemin en revenant au point de départ
    if cycle and len(ordered) > 0:
        ordered.append(ordered[0])

    return np.array(ordered, dtype=int)


def distance_point_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Distance between a point p and the segment [a, b]."""
    p = np.asarray(p, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    ab = b - a
    denom = np.dot(ab, ab)
    if denom == 0:  # a and b are the same point
        return float(np.linalg.norm(p - a))

    ap = p - a
    t = np.dot(ap, ab) / denom
    t = max(0.0, min(1.0, t))
    projection = a + t * ab
    return float(np.linalg.norm(p - projection))


def rdp(points: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
    """
    Douglas–Peucker polyline simplification.
    """
    points = np.asarray(points, dtype=float)
    if len(points) <= 2:
        return points.copy()

    start, end = points[0], points[-1]

    max_dist = 0.0
    index = -1
    for i in range(1, len(points) - 1):
        d = distance_point_segment(points[i], start, end)
        if d > max_dist:
            max_dist = d
            index = i
    if max_dist > epsilon and index != -1:
        left = rdp(points[: index + 1], epsilon)
        right = rdp(points[index:], epsilon)
        return np.vstack((left[:-1], right))
    else:
        return np.vstack((start, end))


def degree_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Retourne l'angle (en degrés) entre deux vecteurs 2D.
    """
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)

    norm1 = math.hypot(v1[0], v1[1])
    norm2 = math.hypot(v2[0], v2[1])
    if norm1 == 0 or norm2 == 0:
        return 0.0

    dot = float(v1[0] * v2[0] + v1[1] * v2[1])
    cos_angle = dot / (norm1 * norm2)
    # Clamp pour éviter les erreurs numériques
    cos_angle = max(-1.0, min(1.0, cos_angle))
    angle_rad = math.acos(cos_angle)
    return math.degrees(angle_rad)


def detect_peak(
    points: np.ndarray,
    angle_range: Tuple[float, float],
    epsilon: float,
) -> np.ndarray:
    """
    Détecte les points de forte courbure (pics) le long d'une polyligne.
    """

    if len(points) < 3:
        return np.empty((0, 2), dtype=float)

    # Simplification RDP
    points = rdp(points, epsilon)

    if len(points) < 3:
        return np.empty((0, 2), dtype=float)

    # Gestion cycle : si déjà fermé, on duplique le premier point à la fin
    if np.array_equal(points[0], points[-1]):
        points = np.vstack((points, points[0]))

    indices: list[int] = []

    # Recherche des pics
    for i in range(len(points) - 2):
        v1 = points[i + 1] - points[i]
        v2 = points[i + 2] - points[i + 1]

        angle = degree_angle(v1, v2)
        if angle_range[0] <= angle <= angle_range[1]:
            indices.append(i + 1)

    if not indices:
        return np.empty((0, 2), dtype=float)

    return points[indices]


class PeakSegment(SegmentStrategy):
    def __init__(
        self, angle_range: Tuple[float, float], peak_detection: bool, epsilon_rdp: float
    ) -> None:
        super().__init__()
        self.threshold = angle_range  # seuil angulaire en degrés
        self.peak_detection = peak_detection
        self.epsilon_rdp = epsilon_rdp

    def _segment(self, binary: np.ndarray) -> np.ndarray:
        skel = skeletonize(binary, method="lee")
        skel = PeakSegment.del_junction(skel)
        if self.peak_detection:
            skel = PeakSegment.del_peak(skel, self.threshold, self.epsilon_rdp)

        return skel

    @staticmethod
    def get_peak(
        skel: np.ndarray,
        angle_range: Tuple[float, float],
        epsilon: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Retourne les coordonnées (row, col) des points de pic sur tout le squelette.
        """
        labeled, n_labels = label(skel, return_num=True, connectivity=2)

        all_rows: list[int] = []
        all_cols: list[int] = []

        for label_id in range(1, n_labels + 1):
            branch_mask = labeled == label_id
            coords = order_branch(branch_mask)

            if len(coords) < 3:
                continue

            coords_peaks = detect_peak(
                coords,
                angle_range=angle_range,
                epsilon=epsilon,
            )
            if coords_peaks.size == 0:
                continue

            all_rows.extend(coords_peaks[:, 0].astype(int).tolist())
            all_cols.extend(coords_peaks[:, 1].astype(int).tolist())

        if not all_rows:
            return np.array([], dtype=int), np.array([], dtype=int)

        return np.array(all_rows, dtype=int), np.array(all_cols, dtype=int)

    @staticmethod
    def del_peak(skel: np.ndarray, threshold: float, epsilon_rdp: float) -> np.ndarray:
        """
        Remove points detected as curvature peaks.

        Parameters
        ----------
        skel : np.ndarray
            Skeleton image.
        threshold : float
            Angle threshold (in degrees) used to classify a point as a peak.
        """
        rows, cols = PeakSegment.get_peak(skel, threshold, epsilon_rdp)
        if rows.size > 0:  # avoid empty indexing
            skel[rows, cols] = WHITE
        return skel

    @staticmethod
    def del_junction(skel: np.ndarray) -> np.ndarray:
        """
        Remove junction points (degree >= 3) in the skeleton.

        Parameters
        ----------
        skel : np.ndarray
            Skeleton image.

        Returns
        -------
        np.ndarray
            Skeleton with junctions removed.
        """
        neighbors = ndi.convolve(skel.astype(int), KERNEL, mode="constant")
        skel[neighbors >= MIN_JUNCTION_DEGREE] = WHITE
        return skel
