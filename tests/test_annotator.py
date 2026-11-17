import importlib.util
from pathlib import Path
from PIL import Image
from fastapi.testclient import TestClient


def test_label_roundtrip_and_original_preservation(tmp_path):
    path = Path(__file__).resolve().parents[1]/"annotator/main.py"
    spec = importlib.util.spec_from_file_location("crop_annotator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.IMAGE_DIR = tmp_path
    Image.new("RGB", (100,40), "white").save(tmp_path/"sample.png")
    with TestClient(module.app) as client:
        assert client.get("/api/images").json() == [{"name": "sample.png", "label": ""}]
        assert client.put("/api/labels/sample.png", json={"text": "AB12"}).status_code == 200
        assert (tmp_path/"sample.txt").read_text() == "AB12"
        assert client.get("/images/sample.png").status_code == 200
        assert client.put("/api/labels/sample.png", json={"text": " "}).status_code == 200
        assert (tmp_path/"sample.txt").read_text() == " "
        assert client.put("/api/labels/sample.png", json={"text": ""}).status_code == 422
        assert client.put("/api/labels/sample.png", json={"text": "A\nB"}).status_code == 422
    assert (tmp_path/"sample.png").exists()
