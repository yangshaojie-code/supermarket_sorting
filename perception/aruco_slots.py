"""Map supermarket shelf ArUco IDs 0-44 onto fixed A-E / L1-L3 / C1-C3 slots."""

from __future__ import annotations

from dataclasses import dataclass
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


def detect_aruco_markers(bgr, dictionary_name: str = ARUCO_DICTIONARY) -> List[MarkerDetection]:
    """Return shelf markers in a BGR image. Requires opencv-contrib aruco."""
    import cv2
    import numpy as np

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    if hasattr(cv2.aruco, "DetectorParameters"):
        parameters = cv2.aruco.DetectorParameters()
    else:
        parameters = cv2.aruco.DetectorParameters_create()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if hasattr(cv2.aruco, "ArucoDetector"):
        corners, ids, _ = cv2.aruco.ArucoDetector(dictionary, parameters).detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
    detections = []
    if ids is None:
        return detections
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        marker_id = int(marker_id)
        if marker_id not in VALID_MARKER_IDS:
            continue
        pts = np.asarray(marker_corners).reshape(4, 2)
        detections.append(
            MarkerDetection(
                marker_id=marker_id,
                slot=slot_from_marker_id(marker_id),
                u=float(pts[:, 0].mean()),
                v=float(pts[:, 1].mean()),
                corners=tuple((float(x), float(y)) for x, y in pts),
            )
        )
    detections.sort(key=lambda item: item.marker_id)
    return detections


def associate_products_to_slots(
    products: Sequence[ProductDetection],
    markers: Sequence[MarkerDetection],
    max_pixel_dist: float = 160.0,
) -> List[dict]:
    """Bind a product bbox to the nearest ArUco, preferring the marker below it.

    Shelf markers sit under the slot, so in a head-camera view the marker
    usually has a larger image v than the product center.
    """
    assignments = []
    used = set()
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
            # Prefer markers below the product (positive dy). Above is penalized.
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
