class Detector:
    """Minimal detector stub."""

    def __init__(self, model_path=None):
        self.model_path = model_path

    def predict(self, image):
        """Return an empty list of detections by default."""
        return []
