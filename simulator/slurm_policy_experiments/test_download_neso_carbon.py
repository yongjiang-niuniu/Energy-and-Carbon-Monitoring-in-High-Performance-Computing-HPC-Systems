import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("download_neso_carbon.py")
SPEC = importlib.util.spec_from_file_location("download_neso_carbon", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class NesoCarbonTests(unittest.TestCase):
    def test_regional_payload_is_normalized(self):
        payload = {
            "data": {
                "regionid": 5,
                "dnoregion": "NPG Yorkshire",
                "shortname": "Yorkshire",
                "data": [
                    {
                        "from": "2025-06-10T20:00Z",
                        "to": "2025-06-10T20:30Z",
                        "intensity": {"forecast": 233, "index": "high"},
                        "generationmix": [{"fuel": "wind", "perc": 3.3}],
                    }
                ],
            }
        }
        frame, metadata = MODULE.normalize_payload(payload)
        self.assertEqual(frame.loc[0, "intensity_gco2_per_kwh"], 233)
        self.assertEqual(frame.loc[0, "generation_wind_pct"], 3.3)
        self.assertEqual(metadata["region_id"], 5)
        self.assertIn("forecast", metadata["value_type"])


if __name__ == "__main__":
    unittest.main()
