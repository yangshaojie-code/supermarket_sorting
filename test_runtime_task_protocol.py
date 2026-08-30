import unittest

from runtime.task_protocol import TaskProtocolError, parse_task_payload, unknown_kinds


SAMPLE = """
{
  "schema_version": 1,
  "run_prefix": "run_a1b2c3d4e5f6",
  "count": 5,
  "targets": [
    {"id": "item_run_a1b2c3d4e5f6_01", "kind": "kele"},
    {"id": "item_run_a1b2c3d4e5f6_02", "kind": "pingguo"},
    {"id": "item_run_a1b2c3d4e5f6_03", "kind": "chengzi"},
    {"id": "item_run_a1b2c3d4e5f6_04", "kind": "zhijin"},
    {"id": "item_run_a1b2c3d4e5f6_05", "kind": "shupian"}
  ]
}
"""


class TaskProtocolTests(unittest.TestCase):
    def test_parses_official_five_order_message(self):
        task = parse_task_payload(SAMPLE)
        self.assertEqual(task.schema_version, 1)
        self.assertEqual(task.run_prefix, "run_a1b2c3d4e5f6")
        self.assertEqual(task.count, 5)
        self.assertEqual(task.kinds(), ("kele", "pingguo", "chengzi", "zhijin", "shupian"))
        self.assertEqual(unknown_kinds(task), ())

    def test_ignores_unpublished_location_fields(self):
        payload = {
            "schema_version": 1,
            "run_prefix": "run_x",
            "count": 1,
            "targets": [{
                "id": "item_run_x_01",
                "kind": "kele",
                "location_id": "D-L2-C2",
                "aruco_id": 32,
                "place_world": [1, 2, 3],
            }],
        }
        task = parse_task_payload(payload)
        self.assertEqual(task.targets[0].kind, "kele")
        self.assertFalse(hasattr(task.targets[0], "location_id"))

    def test_rejects_count_mismatch_and_missing_kind(self):
        with self.assertRaises(TaskProtocolError):
            parse_task_payload({
                "schema_version": 1,
                "run_prefix": "run_x",
                "count": 2,
                "targets": [{"id": "a", "kind": "kele"}],
            })
        with self.assertRaises(TaskProtocolError):
            parse_task_payload({
                "schema_version": 1,
                "run_prefix": "run_x",
                "targets": [{"id": "a"}],
            })


if __name__ == "__main__":
    unittest.main()
