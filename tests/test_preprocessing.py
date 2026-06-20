from vision_pipeline.preprocessing.transforms import resize, normalize


def test_resize_returns_size():
    img = "fake-image"
    out = resize(img, (100, 200))
    assert out["original"] == img
    assert out["size"] == (100, 200)


def test_normalize_flag():
    img = "fake-image"
    out = normalize(img)
    assert out.get("normalized") is True
