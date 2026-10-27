import io
import unittest
from PIL import Image
from fastapi.testclient import TestClient

import sys
import os

# Ensure backend directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from app.main import app

class TestSatQueryAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def create_test_image_bytes(self):
        """Create a simple test RGB image in memory."""
        img = Image.new("RGB", (200, 200), color=(50, 100, 150))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["engine"], "Qwen/Qwen2.5-VL-3B-Instruct")
        self.assertIn("model_status", data)

    def test_model_status_endpoint(self):
        response = self.client.get("/model/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn(data["status"], ["unloaded", "loading", "ready", "error"])
        self.assertEqual(data["model_id"], "Qwen/Qwen2.5-VL-3B-Instruct")

    def test_analyze_empty_question(self):
        img_bytes = self.create_test_image_bytes()
        response = self.client.post(
            "/analyze",
            files={"image": ("test_sat.png", img_bytes, "image/png")},
            data={"question": "   "}
        )
        self.assertEqual(response.status_code, 400)

    def test_analyze_response_schema(self):
        img_bytes = self.create_test_image_bytes()
        response = self.client.post(
            "/analyze",
            files={"image": ("test_sat.png", img_bytes, "image/png")},
            data={"question": "What objects are visible in this image?"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Verify exact required schema
        self.assertIn(data["status"], ["answered", "loading", "error"])
        self.assertIsNone(data["evidence_score"])
        self.assertIsInstance(data["objects"], list)
        self.assertEqual(len(data["objects"]), 0)


if __name__ == "__main__":
    unittest.main()
