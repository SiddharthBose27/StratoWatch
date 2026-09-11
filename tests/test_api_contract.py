import inspect
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.main import app, health, multi_site_run, single_site_run  # noqa: E402


class ApiContractTests(unittest.TestCase):
    def test_health_response(self):
        self.assertEqual(health(), {"ok": True})

    def test_expected_routes_exist(self):
        routes = {
            (method, route.path)
            for route in app.routes
            for method in getattr(route, "methods", set())
        }
        self.assertIn(("GET", "/health"), routes)
        self.assertIn(("POST", "/api/single-site/run"), routes)
        self.assertIn(("POST", "/api/multi-site/run"), routes)

    def test_request_parameter_names(self):
        self.assertEqual(set(inspect.signature(single_site_run).parameters), {"file", "model_name"})
        self.assertEqual(
            set(inspect.signature(multi_site_run).parameters),
            {"site_count", "model_name", "files"},
        )

    def test_response_contract_is_visible_in_runners(self):
        runner = (ROOT / "backend" / "app" / "runner.py").read_text()
        multi_runner = (ROOT / "backend" / "app" / "multi_runner.py").read_text()
        for source in [runner, multi_runner]:
            self.assertIn('"metrics"', source)
            self.assertIn('"plots"', source)
        self.assertIn('"warning"', multi_runner)
        self.assertIn('"logs"', runner)


if __name__ == "__main__":
    unittest.main()