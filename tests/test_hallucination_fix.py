import unittest
import os
import sys
from PIL import Image
import io
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
from app.main import app

class TestHallucinationPolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.sample_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "samples", "satellite_test.png"))
        with open(cls.sample_path, "rb") as f:
            cls.sample_bytes = f.read()

        # Pure green image (forest without buildings)
        img = Image.new("RGB", (256, 256), color=(34, 139, 34))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        cls.green_bytes = buf.getvalue()

    def test_category_a_answerable_presence(self):
        """Test A: Answerable presence of buildings."""
        resp = self.client.post("/analyze", data={"question": "Are there buildings visible?"}, files={"image": ("scene.png", self.sample_bytes, "image/png")})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertFalse(data["abstained"])
        self.assertEqual(data["status"], "answered")
        self.assertTrue(len(data["detections"]) > 0)
        self.assertIn("building", data["answer"].lower())

    def test_category_b_answerable_roads(self):
        """Test B: Answerable presence of roads."""
        resp = self.client.post("/analyze", data={"question": "Are there roads visible?"}, files={"image": ("scene.png", self.sample_bytes, "image/png")})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertFalse(data["abstained"])
        self.assertEqual(data["status"], "answered")
        self.assertTrue(len(data["detections"]) > 0)

    def test_category_c_impossible_temporal(self):
        """Test C: Impossible temporal question must be refused."""
        resp = self.client.post("/analyze", data={"question": "How many buildings were constructed last year?"}, files={"image": ("scene.png", self.sample_bytes, "image/png")})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["abstained"])
        self.assertEqual(data["status"], "abstained")
        self.assertEqual(len(data["detections"]), 0)
        self.assertIn("single satellite image", data["answer"].lower())

    def test_category_d_unsupported_sar(self):
        """Test D: Unsupported SAR/spectral query on RGB must be refused."""
        resp = self.client.post("/analyze", data={"question": "What is the SAR backscatter value?"}, files={"image": ("scene.png", self.sample_bytes, "image/png")})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["abstained"])
        self.assertEqual(data["status"], "abstained")
        self.assertEqual(len(data["detections"]), 0)
        self.assertIn("rgb imagery", data["answer"].lower())

    def test_category_e_unsupported_population(self):
        """Test E: Exact population / demographic query must be refused."""
        resp = self.client.post("/analyze", data={"question": "What is the exact population of this area?"}, files={"image": ("scene.png", self.sample_bytes, "image/png")})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["abstained"])
        self.assertEqual(data["status"], "abstained")
        self.assertEqual(len(data["detections"]), 0)
        self.assertIn("population", data["answer"].lower())

    def test_category_f_ambiguous_absent_object(self):
        """Test F: Count of absent/ambiguous object must refuse, not guess."""
        resp = self.client.post("/analyze", data={"question": "How many airports are visible?"}, files={"image": ("scene.png", self.sample_bytes, "image/png")})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["abstained"])
        self.assertEqual(data["status"], "abstained")
        self.assertEqual(len(data["detections"]), 0)

    def test_category_g_instance_counting(self):
        """Test G: Counting only answers when verified, never raw proposal dump."""
        resp = self.client.post("/analyze", data={"question": "How many buildings are visible?"}, files={"image": ("scene.png", self.sample_bytes, "image/png")})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertFalse(data["abstained"])
        self.assertIn("validated", data["answer"].lower())

    def test_extra_unobservable_city_name(self):
        """Extra: City/geographic name request must be refused."""
        resp = self.client.post("/analyze", data={"question": "What is the name of this city?"}, files={"image": ("scene.png", self.sample_bytes, "image/png")})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["abstained"])
        self.assertIn("geographic names", data["answer"].lower())

    def test_extra_unobservable_ownership(self):
        """Extra: Company ownership request must be refused."""
        resp = self.client.post("/analyze", data={"question": "Which building belongs to company X?"}, files={"image": ("scene.png", self.sample_bytes, "image/png")})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["abstained"])
        self.assertIn("ownership", data["answer"].lower())

    def test_extra_spatial_missing_anchor(self):
        """Extra: Spatial query with missing anchor must refuse."""
        resp = self.client.post("/analyze", data={"question": "What is north of the airport?"}, files={"image": ("scene.png", self.sample_bytes, "image/png")})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["abstained"])

    def test_absence_case(self):
        """Extra: Presence query on negative scene cleanly states absence."""
        resp = self.client.post("/analyze", data={"question": "Are there buildings visible?"}, files={"image": ("green.png", self.green_bytes, "image/png")})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["abstained"])
        self.assertEqual(len(data["detections"]), 0)


if __name__ == "__main__":
    unittest.main()
