import unittest

from runtime.ros_contract import REQUIRED_JOINT_NAMES, topic_contract


class RosContractTests(unittest.TestCase):
    def test_retail_topics_not_tourism_instruction(self):
        contract = topic_contract()
        self.assertEqual(contract["task"], "/supermarket_sorting/task")
        self.assertEqual(contract["scan"], "/slamware_ros_sdk_server_node/scan")
        self.assertEqual(contract["control"]["cmd_vel"], "/cmd_vel")
        self.assertNotIn("/material/instruction", contract.values())
        self.assertEqual(len(REQUIRED_JOINT_NAMES), 17)


if __name__ == "__main__":
    unittest.main()
