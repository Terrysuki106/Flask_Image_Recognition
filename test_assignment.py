"""Test application responses and image preprocessing."""

from io import BytesIO
import re

import pytest
from PIL import Image

import app as app_module
from model import preprocess_img


def make_image(mode="RGB"):
    """Create an in-memory PNG image for testing."""
    image = Image.new(mode, (32, 32), color="white")
    stream = BytesIO()
    image.save(stream, format="PNG")
    stream.seek(0)
    return stream


def test_home_page(client):
    """The home page provides an upload form."""
    response = client.get("/")

    assert response.status_code == 200
    assert b'type="file"' in response.data


def test_valid_upload(client, monkeypatch):
    """A valid image displays the predicted digit."""
    calls = []

    def fake_predict(image):
        calls.append(image.shape)
        return 7

    monkeypatch.setattr(app_module, "predict_result", fake_predict)

    response = client.post(
        "/prediction",
        data={"file": (make_image(), "digit.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert calls == [(1, 224, 224, 3)]
    assert re.search(rb">\s*7\s*<", response.data)


def test_missing_upload(client):
    """A missing file displays the application's error message."""
    response = client.post(
        "/prediction",
        data={},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"File cannot be processed." in response.data


def test_invalid_upload(client, monkeypatch):
    """Invalid image bytes are rejected before prediction."""
    def unexpected_prediction(_image):
        pytest.fail("Prediction must not run for invalid image data.")

    monkeypatch.setattr(
        app_module, "predict_result", unexpected_prediction
    )

    response = client.post(
        "/prediction",
        data={"file": (BytesIO(b"not an image"), "invalid.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"File cannot be processed." in response.data


def test_prediction_rejects_get(client):
    """The prediction route accepts POST requests only."""
    response = client.get("/prediction")

    assert response.status_code == 405


@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA"])
def test_preprocessing_image_modes(mode):
    """Supported image modes produce normalized three-channel batches."""
    processed = preprocess_img(make_image(mode))

    assert processed.shape == (1, 224, 224, 3)
    assert processed.min() >= 0
    assert processed.max() <= 1
    assert processed == pytest.approx(1.0)
