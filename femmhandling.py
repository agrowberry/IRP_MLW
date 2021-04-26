import femm
import numpy as np
import plotly.graph_objects as go
import datahandling


class Preprocessor:
    def __init__(self):
        self.gap_width = None
        self.gap_height = None
        self.core_width = None
        self.inner_radius = None
        self.outer_radius = None
        self.core_radius = None

    def make_back_iron(self, dimensions=None):
        if dimensions is None:
            dimension_dict = datahandling.start_up()
        self.gap_width = dimension_dict['']
        self.hello = None

