import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "analyse_node_inventory.py"
SPEC = importlib.util.spec_from_file_location("analyse_node_inventory", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class NodeInventoryHelpersTest(unittest.TestCase):
    def test_expand_hostlist_range_and_list(self) -> None:
        self.assertEqual(
            MODULE.expand_hostlist("node[003-005,010]"),
            ["node003", "node004", "node005", "node010"],
        )

    def test_expand_multiple_top_level_groups(self) -> None:
        self.assertEqual(
            MODULE.expand_hostlist("node001,gpu[21-22]"),
            ["node001", "gpu21", "gpu22"],
        )

    def test_inferred_groups_separate_published_spec_from_mapping(self) -> None:
        group, hardware, evidence = MODULE.inferred_group("gpu23")
        self.assertEqual(group, "gpu_21_26")
        self.assertIn("H100", hardware)
        self.assertIn("official specification", hardware)
        self.assertIn("mapping inferred", evidence)

    def test_unknown_node_is_not_guessed(self) -> None:
        group, _, evidence = MODULE.inferred_group("special01")
        self.assertEqual(group, "unclassified")
        self.assertIn("confirmation", evidence)


if __name__ == "__main__":
    unittest.main()
