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
        femm.openfemm()
        femm.newdocument(0)
        femm.mi_probdef(0, 'millimeters', 'axi', 1.e-8, 0, 30)

    def build_back_iron(self, dimensions=None):
        if dimensions is None:
            dimension_dict = datahandling.start_up()
        self.gap_width = dimension_dict['']

    def build_coil(self, coil_dict):
        """
        load coil_cross sections into femm environment
        :param coil_dict:
        :return:
        """
        for section, section_tuple in coil_dict.items():
            path_point = np.array(section_tuple[0])
            profile = section_tuple[1] + path_point
            # draw line segments between nodes
            for index, point in enumerate(profile):
                last_point = profile[index - 1]
                femm.mi_drawline(last_point[1], last_point[0], point[1], point[0])
            femm.mi_addblocklabel(path_point[1], path_point[0])
            femm.mi_selectlabel(path_point[1], path_point[0])
            femm.mi_setblockprop(str(section))
            femm.mi_clearselected()






