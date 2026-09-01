"""Map supermarket shelf ArUco IDs 0-44 onto fixed A-E / L1-L3 / C1-C3 slots."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import List, Sequence, Tuple

SHELVES = ("A", "B", "C", "D", "E")
VALID_MARKER_IDS = range(45)
MARKER_SIZE_M = 0.03
ARUCO_DICTIONARY = "DICT_4X4_50"


@dataclass(frozen=True)
class Slot:
    marker_id: int
    shelf: str
    layer: int
    column: int

    @property
    def name(self) -> str:
        return f"{self.shelf}-L{self.layer}-C{self.column}"


@dataclass(frozen=True)
class MarkerDetection:
    marker_id: int
    slot: Slot
    u: float
    v: float
    corners: Tuple[Tuple[float, float], ...] = ()
    inferred: bool = False


@dataclass(frozen=True)
class ProductDetection:
    kind: str
    u: float
    v: float
    conf: float
    w: float = 0.0
    h: float = 0.0


def slot_from_marker_id(marker_id: int) -> Slot:
    marker_id = int(marker_id)
    if marker_id not in VALID_MARKER_IDS:
        raise ValueError(f"ArUco id {marker_id} is not a shelf marker (0-44)")
    shelf = SHELVES[marker_id // 9]
    local = marker_id % 9
    layer = local // 3 + 1
    column = local % 3 + 1
    return Slot(marker_id=marker_id, shelf=shelf, layer=layer, column=column)


def _aruco_parameters():
    import cv2

    if hasattr(cv2.aruco, "DetectorParameters"):
        parameters = cv2.aruco.DetectorParameters()
    else:
        parameters = cv2.aruco.DetectorParameters_create()
    parameters.minMarkerPerimeterRate = 0.01
    parameters.maxMarkerPerimeterRate = 4.0
    parameters.minDistanceToBorder = 0
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 23
    parameters.adaptiveThreshConstant = 7
    parameters.perspectiveRemovePixelPerCell = 8
    if hasattr(parameters, "detectInvertedMarker"):
        parameters.detectInvertedMarker = True
    refine = getattr(cv2.aruco, "CORNER_REFINE_SUBPIX", None)
    if refine is not None:
        parameters.cornerRefinementMethod = refine
    return parameters


def _run_aruco_detector(gray, dictionary, parameters):
    import cv2

    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, parameters).detectMarkers(gray)
    return cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)


def _gray_variants(gray):
    import cv2

    variants = [(gray, 1.0)]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    variants.append((clahe, 1.0))
    variants.append((cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC), 2.0))
    variants.append((cv2.bitwise_not(gray), 1.0))
    return variants


def detect_aruco_markers(bgr, dictionary_name: str = ARUCO_DICTIONARY) -> List[MarkerDetection]:
    """Return shelf markers in a BGR image. Requires opencv-contrib aruco."""
    import cv2
    import numpy as np

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    parameters = _aruco_parameters()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    by_id = {}
    for image, scale in _gray_variants(gray):
        corners, ids, _rejected = _run_aruco_detector(image, dictionary, parameters)
        if ids is None:
            continue
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            marker_id = int(marker_id)
            if marker_id not in VALID_MARKER_IDS or marker_id in by_id:
                continue
            pts = np.asarray(marker_corners).reshape(4, 2) / float(scale)
            by_id[marker_id] = MarkerDetection(
                marker_id=marker_id,
                slot=slot_from_marker_id(marker_id),
                u=float(pts[:, 0].mean()),
                v=float(pts[:, 1].mean()),
                corners=tuple((float(x), float(y)) for x, y in pts),
            )
    detections = sorted(by_id.values(), key=lambda item: item.marker_id)
    return detections


def detect_marker_quads(bgr, dictionary_name: str = ARUCO_DICTIONARY) -> List[List[List[float]]]:
    """Decoded plus rejected quads, for GS=0 frames where IDs fail to decode."""
    import cv2
    import numpy as np

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    parameters = _aruco_parameters()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    quads = []
    seen = []
    for image, scale in ((gray, 1.0), (cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC), 2.0)):
        corners, _ids, rejected = _run_aruco_detector(image, dictionary, parameters)
        groups = []
        if corners:
            groups.extend(corners)
        if rejected is not None:
            groups.extend(rejected)
        for item in groups:
            pts = (np.asarray(item).reshape(4, 2) / float(scale)).tolist()
            center = (sum(p[0] for p in pts) / 4.0, sum(p[1] for p in pts) / 4.0)
            if any(abs(center[0] - other[0]) < 12 and abs(center[1] - other[1]) < 12 for other in seen):
                continue
            width = max(p[0] for p in pts) - min(p[0] for p in pts)
            height = max(p[1] for p in pts) - min(p[1] for p in pts)
            if width < 12 or height < 12 or max(width, height) / max(min(width, height), 1.0) > 2.5:
                continue
            seen.append(center)
            quads.append([[float(x), float(y)] for x, y in pts])
    return quads


def _quad_center(quad) -> Tuple[float, float]:
    return (
        sum(point[0] for point in quad) / 4.0,
        sum(point[1] for point in quad) / 4.0,
    )


def _lattice_pitches(markers: Sequence[MarkerDetection]) -> Tuple[float, float]:
    column_spacings = []
    layer_spacings = []
    for left, right in combinations(markers, 2):
        if left.slot.shelf != right.slot.shelf:
            continue
        dcol = right.slot.column - left.slot.column
        dlayer = right.slot.layer - left.slot.layer
        if dlayer == 0 and dcol != 0:
            pitch = (right.u - left.u) / dcol
            if pitch > 40.0:
                column_spacings.append(pitch)
        if dcol == 0 and dlayer != 0:
            pitch = (right.v - left.v) / dlayer
            if pitch > 40.0:
                layer_spacings.append(pitch)
    column_pitch = (
        sorted(column_spacings)[len(column_spacings) // 2] if column_spacings else 220.0
    )
    layer_pitch = (
        sorted(layer_spacings)[len(layer_spacings) // 2] if layer_spacings else 220.0
    )
    return float(column_pitch), float(layer_pitch)


def dedupe_markers_by_pixel(
    markers: Sequence[MarkerDetection],
    min_sep: float = 14.0,
) -> List[MarkerDetection]:
    """Drop a second decode that landed on the same square (e.g. 37 on top of 28)."""
    kept: List[MarkerDetection] = []
    for item in markers:
        if any(abs(item.u - other.u) < min_sep and abs(item.v - other.v) < min_sep for other in kept):
            continue
        kept.append(item)
    return kept


def infer_markers_from_quads(
    decoded: Sequence[MarkerDetection],
    quads: Sequence[Sequence[Sequence[float]]],
) -> List[MarkerDetection]:
    """Assign IDs to undecoded squares using the 3x3 slot lattice.

    Score every quad against every decoded anchor, then greedily take the
    lowest-residual (id, quad) pairs so a left-edge C1 square cannot steal
    D-L2-C2 from the square sitting under the product.
    """
    decoded = dedupe_markers_by_pixel(decoded)
    if not decoded or not quads:
        return []
    column_pitch, layer_pitch = _lattice_pitches(decoded)
    used_ids = {item.marker_id for item in decoded}
    candidates = []
    for quad in quads:
        u, v = _quad_center(quad)
        if any(abs(u - item.u) < 14 and abs(v - item.v) < 14 for item in decoded):
            continue
        best = None
        for anchor in decoded:
            dcol = int(round((u - anchor.u) / column_pitch))
            dlayer = int(round((v - anchor.v) / layer_pitch))
            column = anchor.slot.column + dcol
            layer = anchor.slot.layer + dlayer
            if not (1 <= column <= 3 and 1 <= layer <= 3):
                continue
            residual_u = abs((u - anchor.u) - dcol * column_pitch)
            residual_v = abs((v - anchor.v) - dlayer * layer_pitch)
            if residual_u > column_pitch * 0.38 or residual_v > layer_pitch * 0.38:
                continue
            marker_id = SHELVES.index(anchor.slot.shelf) * 9 + (layer - 1) * 3 + (column - 1)
            residual = residual_u + residual_v
            if best is None or residual < best[0]:
                best = (residual, marker_id)
        if best is None:
            continue
        candidates.append((best[0], best[1], u, v, quad))
    candidates.sort(key=lambda item: item[0])
    inferred = []
    claimed_quads = set()
    for _residual, marker_id, u, v, quad in candidates:
        if marker_id in used_ids:
            continue
        key = (round(u, 1), round(v, 1))
        if key in claimed_quads:
            continue
        used_ids.add(marker_id)
        claimed_quads.add(key)
        inferred.append(
            MarkerDetection(
                marker_id=marker_id,
                slot=slot_from_marker_id(marker_id),
                u=float(u),
                v=float(v),
                corners=tuple((float(x), float(y)) for x, y in quad),
                inferred=True,
            )
        )
    return inferred


def associate_products_to_slots(
    products: Sequence[ProductDetection],
    markers: Sequence[MarkerDetection],
    max_pixel_dist: float = 220.0,
) -> List[dict]:
    """Bind a product bbox to the nearest ArUco on the same column, below it.

    Shelf markers sit on the front lip of the slot, so in a head-camera view the
    marker usually has a larger image v than the product center. A marker on
    the layer above, or a full column to the side, is a different slot.
    """
    assignments = []
    used = set()
    column_pitch, layer_pitch = _lattice_pitches(markers) if len(markers) >= 2 else (220.0, 220.0)
    max_dx = max(120.0, 0.48 * column_pitch)
    max_dy = max(160.0, 1.15 * layer_pitch)
    for product in products:
        best = None
        best_score = None
        for marker in markers:
            if marker.marker_id in used:
                continue
            dx = marker.u - product.u
            dy = marker.v - product.v
            dist = (dx * dx + dy * dy) ** 0.5
            if dist > max_pixel_dist:
                continue
            if abs(dx) > max_dx:
                continue
            # Marker on the shelf above the product (smaller v) is a different layer.
            if dy < -20.0 or dy > max_dy:
                continue
            score = dist + (80.0 if dy < 0 else 0.0)
            if best_score is None or score < best_score:
                best, best_score = marker, score
        if best is None:
            assignments.append({
                "kind": product.kind,
                "conf": product.conf,
                "pixel": [product.u, product.v],
                "slot": None,
                "reason": "no nearby ArUco",
            })
            continue
        used.add(best.marker_id)
        assignments.append({
            "kind": product.kind,
            "conf": product.conf,
            "pixel": [product.u, product.v],
            "slot": best.slot.name,
            "marker_id": best.marker_id,
            "marker_pixel": [best.u, best.v],
            "inferred_marker": bool(best.inferred),
        })
    return assignments


class KindSlotMap:
    """Session map: product kind -> shelf slots observed this run."""

    def __init__(self):
        self._slots: dict[str, dict[str, dict]] = {}

    def observe(self, kind: str, slot: Slot, confidence: float = 0.0) -> None:
        by_kind = self._slots.setdefault(kind, {})
        previous = by_kind.get(slot.name)
        if previous is None or confidence >= previous["confidence"]:
            by_kind[slot.name] = {
                "slot": slot.name,
                "marker_id": slot.marker_id,
                "shelf": slot.shelf,
                "layer": slot.layer,
                "column": slot.column,
                "confidence": float(confidence),
            }

    def lookup(self, kind: str) -> List[dict]:
        return list(self._slots.get(kind, {}).values())

    def as_dict(self) -> dict:
        return {kind: list(slots.values()) for kind, slots in self._slots.items()}
