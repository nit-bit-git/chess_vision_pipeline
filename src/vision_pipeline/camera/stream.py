class CameraStream:
    """Simple camera stream stub for tests and local runs."""

    def __init__(self, src=0):
        self.src = src

    def read_frame(self):
        """Return a placeholder frame (None by default)."""
        return None
