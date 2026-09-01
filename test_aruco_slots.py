import unittest

from perception.aruco_slots import (
    KindSlotMap,
    MarkerDetection,
    ProductDetection,
    associate_products_to_slots,
    detect_aruco_markers,
    infer_markers_from_quads,
    slot_from_marker_id,
)


class ArucoSlotTests(unittest.TestCase):
    def test_official_d_shelf_layout(self):
        self.assertEqual(slot_from_marker_id(0).name, "A-L1-C1")
        self.assertEqual(slot_from_marker_id(8).name, "A-L3-C3")
        self.assertEqual(slot_from_marker_id(27).name, "D-L1-C1")
        self.assertEqual(slot_from_marker_id(31).name, "D-L2-C2")
        self.assertEqual(slot_from_marker_id(32).name, "D-L2-C3")
        self.assertEqual(slot_from_marker_id(44).name, "E-L3-C3")

    def test_rejects_ids_outside_0_44(self):
        with self.assertRaises(ValueError):
            slot_from_marker_id(45)

    def test_associates_product_to_marker_below_it(self):
        markers = [
            MarkerDetection(31, slot_from_marker_id(31), u=320.0, v=300.0),
            MarkerDetection(32, slot_from_marker_id(32), u=400.0, v=300.0),
        ]
        products = [ProductDetection("kele", u=318.0, v=220.0, conf=0.9)]
        assigned = associate_products_to_slots(products, markers)
        self.assertEqual(assigned[0]["slot"], "D-L2-C2")
        self.assertEqual(assigned[0]["marker_id"], 31)

    def test_does_not_bind_product_to_the_shelf_above_it(self):
        # Head looking at L1: marker 28 above a cola that sits on L2.
        markers = [MarkerDetection(28, slot_from_marker_id(28), u=181.0, v=331.0)]
        products = [ProductDetection("kele", u=217.0, v=444.0, conf=0.94)]
        assigned = associate_products_to_slots(products, markers)
        self.assertIsNone(assigned[0]["slot"])
        self.assertEqual(assigned[0]["reason"], "no nearby ArUco")

    def test_does_not_bind_across_columns_to_marker_32(self):
        markers = [
            MarkerDetection(28, slot_from_marker_id(28), u=158.0, v=144.3),
            MarkerDetection(29, slot_from_marker_id(29), u=452.0, v=124.4),
            MarkerDetection(32, slot_from_marker_id(32), u=403.0, v=379.9),
        ]
        products = [ProductDetection("kele", u=211.0, v=298.0, conf=0.95)]
        assigned = associate_products_to_slots(products, markers)
        self.assertIsNone(assigned[0]["slot"])

    def test_infers_undecoded_l2_c2_and_binds_the_can(self):
        decoded = [
            MarkerDetection(28, slot_from_marker_id(28), u=158.0, v=144.3),
            MarkerDetection(29, slot_from_marker_id(29), u=452.0, v=124.4),
            MarkerDetection(32, slot_from_marker_id(32), u=403.0, v=379.9),
        ]
        quad_31 = ((211.0, 393.0), (235.0, 390.0), (235.0, 409.0), (214.0, 412.0))
        inferred = infer_markers_from_quads(decoded, [quad_31])
        self.assertEqual([item.marker_id for item in inferred], [31])
        assigned = associate_products_to_slots(
            [ProductDetection("kele", u=211.0, v=298.0, conf=0.95)],
            list(decoded) + inferred,
        )
        self.assertEqual(assigned[0]["slot"], "D-L2-C2")
        self.assertEqual(assigned[0]["marker_id"], 31)
        self.assertTrue(assigned[0]["inferred_marker"])

    def test_c1_quad_does_not_steal_l2_c2_from_the_square_under_the_can(self):
        decoded = [
            MarkerDetection(28, slot_from_marker_id(28), u=154.0, v=129.5),
            MarkerDetection(29, slot_from_marker_id(29), u=452.0, v=108.2),
            MarkerDetection(32, slot_from_marker_id(32), u=401.8, v=367.0),
            MarkerDetection(37, slot_from_marker_id(37), u=153.75, v=129.5),
        ]
        quads = [
            ((131.0, 112.0), (170.0, 109.0), (176.0, 147.0), (139.0, 150.0)),
            ((391.0, 359.0), (413.0, 357.0), (412.0, 375.0), (391.0, 377.0)),
            ((30.0, 401.0), (36.0, 421.0), (12.0, 424.0), (7.0, 404.0)),
            ((233.0, 378.0), (233.0, 397.0), (212.0, 398.0), (210.0, 380.0)),
        ]
        inferred = infer_markers_from_quads(decoded, quads)
        by_id = {item.marker_id: item for item in inferred}
        self.assertIn(31, by_id)
        self.assertGreater(by_id[31].u, 180.0)
        self.assertLess(by_id[31].u, 250.0)
        assigned = associate_products_to_slots(
            [ProductDetection("kele", u=209.0, v=285.0, conf=0.95)],
            list(decoded) + inferred,
        )
        self.assertEqual(assigned[0]["slot"], "D-L2-C2")
        self.assertEqual(assigned[0]["marker_id"], 31)

    def test_kind_slot_map_keeps_best_confidence(self):
        mapping = KindSlotMap()
        mapping.observe("kele", slot_from_marker_id(32), 0.4)
        mapping.observe("kele", slot_from_marker_id(32), 0.8)
        mapping.observe("kele", slot_from_marker_id(31), 0.5)
        slots = mapping.lookup("kele")
        self.assertEqual({item["slot"] for item in slots}, {"D-L2-C2", "D-L2-C3"})
        conf32 = next(item["confidence"] for item in slots if item["marker_id"] == 32)
        self.assertEqual(conf32, 0.8)

    def test_detects_generated_dict_4x4_id_32(self):
        import cv2
        import numpy as np

        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        marker = cv2.aruco.generateImageMarker(dictionary, 32, 80)
        canvas = np.full((240, 320, 3), 240, dtype=np.uint8)
        canvas[80:160, 120:200] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        found = detect_aruco_markers(canvas)
        self.assertEqual([item.marker_id for item in found], [32])
        self.assertEqual(found[0].slot.name, "D-L2-C3")


class P3PreviewNoteTests(unittest.TestCase):
    def test_dark_frame_points_at_missing_background_gs(self):
        from runtime.p3_preview import _empty_marker_note

        note = _empty_marker_note({"rgb_mean": 1.53, "marker_quads": []})
        self.assertIn("SUPERMARKET_GS_NO_BACKGROUND", note)

    def test_visible_quads_without_ids(self):
        from runtime.p3_preview import _empty_marker_note

        note = _empty_marker_note({"rgb_mean": 80.0, "marker_quads": [{"pixel": [1, 2]}]})
        self.assertIn("did not decode", note)

    def test_score_prefers_a_bound_slot_over_extra_upper_markers(self):
        from runtime.p3_preview import _score

        looking_at_l1 = {
            "assignments": [{"slot": None}],
            "products": [1],
            "markers": [1, 2],
            "marker_quads": [],
            "red_blobs": [],
        }
        bound_l2 = {
            "assignments": [{"marker_id": 32, "slot": "D-L2-C3"}],
            "products": [1],
            "markers": [1],
            "marker_quads": [],
            "red_blobs": [],
        }
        self.assertGreater(_score(bound_l2), _score(looking_at_l1))


if __name__ == "__main__":
    unittest.main()
