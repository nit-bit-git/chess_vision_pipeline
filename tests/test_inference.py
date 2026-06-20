from vision_pipeline.inference.detector import Detector


def test_detector_predicts_list():
    det = Detector()
    out = det.predict("fake-image")
    assert isinstance(out, list)
