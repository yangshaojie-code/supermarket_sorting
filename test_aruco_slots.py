import unittest

from perception.aruco_slots import (
    KindSlotMap,
    MarkerDetection,
    ProductDetection,
    associate_products_to_slots,
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

    def test_kind_slot_map_keeps_best_confidence(self):
        mapping = KindSlotMap()
        mapping.observe("kele", slot_from_marker_id(32), 0.4)
        mapping.observe("kele", slot_from_marker_id(32), 0.8)
        mapping.observe("kele", slot_from_marker_id(31), 0.5)
        slots = mapping.lookup("kele")
        self.assertEqual({item["slot"] for item in slots}, {"D-L2-C2", "D-L2-C3"})
        conf32 = next(item["confidence"] for item in slots if item["marker_id"] == 32)
        self.assertEqual(conf32, 0.8)


if __name__ == "__main__":
    unittest.main()
