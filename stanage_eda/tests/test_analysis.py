import math
import unittest

from stanage_eda.scripts.analyse_stanage import parse_tres_value, unique_headers


class AnalysisHelpersTest(unittest.TestCase):
    def test_duplicate_headers_are_stable(self) -> None:
        self.assertEqual(
            unique_headers(["AllocTRES", "AllocTRES", "", "ReqTRES", "ReqTRES"]),
            ["AllocTRES", "AllocTRES__2", "_blank_3", "ReqTRES", "ReqTRES__2"],
        )

    def test_gpu_count_excludes_telemetry(self) -> None:
        value = "gres/gpu:h100=2,gres/gpumem=88046829568,gres/gpuutil=100"
        self.assertEqual(parse_tres_value(value, "gpu"), 2)

    def test_implausible_gpu_count_is_missing(self) -> None:
        self.assertTrue(math.isnan(parse_tres_value("gres/gpu=88046829568", "gpu")))

    def test_memory_is_converted_to_mib(self) -> None:
        self.assertEqual(parse_tres_value("mem=8G", "mem"), 8192)
        self.assertEqual(parse_tres_value("mem=4096M", "mem"), 4096)


if __name__ == "__main__":
    unittest.main()
