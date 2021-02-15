import scipy as sp
import numpy as np
import plotly.graph_objects as go
import pandas as pd


class DataEntry(object):
    def __init__(self):
        self.std_inputs_dict = {'core_length': 0.0,
                                'core_minor_axis': 0.0,
                                'core_major_axis': 0.0,
                                'num_of_turns': 0.0,
                                'outer_minor_axis': 0.0,
                                'outer_major_axis': 0.0,
                                'spacing': 0.0
                                }

        self.yes_answer = ['y', 'Y', 'Yes', 'YES', 'yes']
        self.no_answer = ['n', 'N', 'no', 'NO', 'No']

    def input_data(self, input_dict=False):
        if not input_dict:
            input_dict = self.std_inputs_dict

        for key in input_dict:
            input_dict[key] = float(input('Input ' + str(key) + ': '))

        return input_dict

    def read_data(self, file):
        with open(file) as f:
            lines = f.readlines()
            input_dict = {}
            for line in lines:
                input_dict[line[0]] = line[1]

        return input_dict

    def insert_model(self, file):
        pass


class FourierManipulation(object):
    def __init__(self):
        self.fig = go.Figure()
        self.geom_dict = {}

    def grad(self, vector_array, normalise=True):
        """
        returns either un-normalised or normalised gradient array of an n-dimension array.
        :param vector_array: n-dimensional array of sequential points of
        form [[x1, y1, z1, ...], [x2, y2, z2, ...], ...].
        :param normalise: kwarg to return either a normalised or un-normalised array
        :return: grad array: n-dimensional array of sequential gradients.
        """
        count = 1
        grad_array = []
        for i in vector_array:
            if count == 1:
                last_i = i
                count += 1
            else:
                grad = i - last_i
                grad_array.append(grad)
                count += 1
                last_i = i
        grad_array = grad_array[0:1] + grad_array
        grad_array = np.stack(grad_array)
        if normalise:
            norm_grad_list = []
            for vector in grad_array:
                norm_vector = vector / np.linalg.norm(vector)
                norm_grad_list.append(norm_vector)
            grad_array = np.stack(norm_grad_list)

        return grad_array

    def make_coil(self, geom_dict, N):
        """
        :param: geom_dict, dictionary in std_input_dict format with geometry params.
        :param: N, no. of points in returned array.
        :return: array of points describing square coil around rectangular core.
        """

        geom_df = pd.Dataframe()

    # returns array with fourier series coefficients.
    def make_fourier_series(self, geometry_dict, plot_result=True):
        val_list = []
        for val in geometry_dict:
            val_list.append(geometry_dict[val])


