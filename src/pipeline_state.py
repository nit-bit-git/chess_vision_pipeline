class PipelineState:
    def __init__(self):
        self.warped_image = None
        self.intersections = None
        self.tilt_ratio = None

        self.board_orientation = None
        self.a1_row = None
        self.a1_col = None
        self.flip_rows = None
        self.flip_cols = None

        self.labels = None
        self.pieces = []

        self.board_matrix = None
        self.fen = None