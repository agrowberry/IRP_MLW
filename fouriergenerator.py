import scipy as sp
import numpy as np
import plotly.graph_objects as go
import plotly
import pandas as pd
import datahandling
from plotly.subplots import make_subplots


class GeometryManipulation():
    def __init__(self):
        self.fig = go.Figure()
        self.geom_dict = {}
        self.point_array = np.zeros((3, 1000))
        self.local_spiral = np.zeros((3, 1000))
        self.main_spiral = np.zeros((3, 1000))
        self.grad_array = np.zeros((3, 1000))
        self.coil_sing_spiral = np.zeros((3, 1000))
        self.coil_spiral = np.zeros((3, 1000))

    def make_helix(self, width, height, length, num_turns, n, output=False):
        t = np.linspace(0, 2 * num_turns, n)
        per = 1
        l = 1.57
        x = 1 * (height / np.pi * (
                np.arcsin(np.sin((np.pi / per) * t + l)) + np.arccos(np.cos((np.pi / per) * t + l))) - height / 2)

        y = 1 * (width / np.pi * (
                np.arcsin(np.sin((np.pi / per) * t)) + np.arccos(np.cos((np.pi / per) * t))) - width / 2)

        z = np.linspace(0, length, n)

        helix_array = np.array([x, y, z])
        if output:
            return helix_array
        else:
            self.main_spiral = np.copy(helix_array)

    def grad(self, vector_array, normalise=True, output=False):
        """
        returns either un-normalised or normalised gradient array of an n-dimension array.
        :param vector_array: n-dimensional array of sequential points of
        form [[x1, y1, z1, ...], [x2, y2, z2, ...], ...].
        :param normalise: kwarg to return either a normalised or un-normalised array
        :param output: kwarg to return grad_array as object or to reassign self.grad_array.
        :return grad array: n-dimensional array of sequential gradients.
        """
        vector_array = np.transpose(vector_array)
        df = pd.DataFrame({'x': vector_array[0],
                           'y': vector_array[1],
                           'z': vector_array[2]})
        grad_df = df.diff()
        grad_df.iloc[0] = grad_df.iloc[1]
        grad_array = grad_df.to_numpy()
        if normalise:
            grad_array = grad_array / np.apply_along_axis(np.linalg.norm, 0, np.transpose(grad_array))[:, None]
        if output:
            return grad_array
        else:
            self.grad_array = np.copy(grad_array)

    def map_points(self, unmapped_local_array, normal_array, trans_array, output=False):
        """
        will map point from global origin [0, 0, 0] to translated and rotated points
        :param unmapped_local_array: n-dimensional array of points to be mapped.
        :param normal_array: n-dimensional array of normals to rotation point.
        :param trans_array: n-dimensional array of translated origins.
        :param output: kwarg conditional for making method static.
        :return: mapped_array: n-dimensional array of mapped points.
        """
        mapped_list = []

        def angle(n, p):
            angle_rad = np.arccos((np.dot(n, p)) / (np.linalg.norm(n) * np.linalg.norm(p)))
            return angle_rad

        for index, unmapped_vector in enumerate(unmapped_local_array):
            normal_vector = normal_array[index]
            unmapped_vector = unmapped_local_array[index]
            trans_vector = trans_array[index]
            # if np.array_equal(normal_vector, normal_array[index - 1]) or index == 0:
            if True:
                rot_angle_y = np.pi / 2 - angle(np.array([1, 0]),
                                                np.array([(normal_vector[0] ** 2 + normal_vector[1] ** 2) ** 0.5,
                                                          normal_vector[2]]))
                rot_angle_z = angle(np.array([0, 1]), np.array([normal_vector[1], normal_vector[0]]))
                r_y = np.copy(np.array([np.array([np.cos(rot_angle_y), 0, np.sin(rot_angle_y)]),
                                        np.array([0, 1, 0]),
                                        np.array([-np.sin(rot_angle_y), 0, np.cos(rot_angle_y)])]))
                r_z = np.copy(np.array([np.array([np.cos(rot_angle_z), -np.sin(rot_angle_z), 0]),
                                        np.array([np.sin(rot_angle_z), np.cos(rot_angle_z), 0]),
                                        np.array([0, 0, 1])]))
            mapped_vector = np.copy(np.matmul(r_z, np.matmul(r_y, unmapped_vector)) + trans_vector)
            mapped_list.append(mapped_vector)
            print('mapping points ' + str(index) + ' of ' + str(unmapped_local_array.shape[0]) + ' : ' + str(
                100 * index / unmapped_local_array.shape[0]) + '%', end='\r')
        if output:
            return np.array(mapped_list)
        else:
            self.point_array = np.transpose(np.array(mapped_list))

    def make_coil(self, N, geom_dict=datahandling.start_up(), plot=False, store=True):
        """
        main script calling functions to make a 3D coil of designated size.
        :param store: conditional to store returned array in .json file.
        :type store: boolean
        :param plot: conditional to plot returned array.
        :type plot: boolean
        :param: geom_dict, dictionary in std_input_dict format with geometry params.
        :type geom_dict: dict
        :param N: no. of points in returned array.
        :type N: int
        :return: array of points describing square coil around rectangular core.
        """
        num_turns = int(geom_dict['num_of_turns'])
        coil_height = (geom_dict['core_length'] - (num_turns - 1) * geom_dict['spacing']) / num_turns
        coil_width = geom_dict['outer_spacing'] - 2 * geom_dict['spacing']
        major_width = geom_dict['core_major_axis'] + geom_dict['spacing'] + coil_width / 2
        major_height = geom_dict['core_minor_axis'] + geom_dict['spacing'] + coil_height / 2
        major_length = geom_dict['core_length']

        self.make_helix(major_width, major_height, major_length, num_turns, N)
        self.grad(np.transpose(self.main_spiral))
        self.coil_spiral = self.make_helix(coil_width, coil_height, 0, int(round(N / 200)), N,
                                           output=True)
        self.map_points(np.transpose(self.coil_spiral), self.grad_array, np.transpose(self.main_spiral))
        if plot:
            self.plot_point_array(major_length, major_width, major_height, coil_width, coil_height, store='png')
        if store:
            datahandling.store_coil_points(self.point_array)

    def plot_point_array(self, m_l, m_w, m_h, c_w, c_h, store=None):
        """
        Creates and displays 3d scatter plot of point array.
        :param store: string specifying file format of stored plot. e.g. 'png'
        :type store: basestring
        :param m_l:major_length
        :param m_w:major_width
        :param m_h:major_height
        :param c_w:coil_width
        :param c_h:coil_height
        :return:
        """
        self.fig.add_trace(go.Scatter3d(x=self.point_array[0], y=self.point_array[1], z=self.point_array[2],
                                        mode='lines', name='Coil Surface'))
        self.fig.add_trace(go.Scatter3d(x=self.main_spiral[0], y=self.main_spiral[1], z=self.main_spiral[2],
                                        name='Coil Path'))

        max_main_dim = max(m_l, m_w, m_h)
        max_local_dim = max(c_w, c_h)
        axis_lims = [-(max_main_dim / 2 + max_local_dim + 1), (max_main_dim / 2 + max_local_dim + 1)]
        self.fig.update_layout(scene=dict(xaxis=dict(range=axis_lims),
                                          yaxis=dict(range=axis_lims),
                                          zaxis=dict(range=axis_lims)))
        if store is not None:
            self.fig.write_image('point_array_scatter', format=store)

        self.fig.show()


class FourierManipulation():
    def __init__(self):
        self.fft_points_array = np.zeros((3, 1000))
        self.unaltered_point_array = np.transpose(datahandling.fetch_coil_points())
        self.reconstructed_point_array = np.zeros(self.unaltered_point_array.shape)

    @staticmethod
    def nextpow2(n):
        """
        returns next highest power of 2 (e.g. for n = 5 returns 8.)
        :param n:
        :return:
        """
        power = 0
        while True:
            answer = 2 ** power
            if answer > n:
                break
            power += 1
        return answer

    def fft_sample(self, sample_rate_percentage=0.5, offset=0, output=False):
        """
        Returns the n-dimensional fft of a given n-dimensional array of points. Down samples and offsets the
        transformed array.

        :param sample_rate_percentage: the size ratio of the returned array (new size = old size * sample_rate_percentage)
        :param offset: the no. of points by which the sampling is shifted.
        :param output: conditional to make method static.
        :return self.fft_points_array: transformed and down sampled nd-array of fourier freq. coefficients.
        """
        # no. of items from list to be removed.
        self.fft_points_array = sp.fft.fftn(self.unaltered_point_array)
        down_sample_num = int(len(self.fft_points_array) * (1 - sample_rate_percentage))
        deletable_indices = []
        # assigns the indices of the array to be removed (appended to deletable_indices)
        for i in range(down_sample_num):
            # finds equally spaced points in the index space [***-***-***].
            deletable_indices.append(int(round(((i + 0.5) / down_sample_num) * len(self.fft_points_array)) + offset))
        # creates list of indices to sample the unaltered_point_array.
        sampled_indices = list(set(range(len(self.fft_points_array))) - set(deletable_indices))
        # maps the list of indices to be sampled to an nd array of sampled points.
        sampled_fft_point_array = np.transpose([*map(self.fft_points_array.__getitem__, sampled_indices)])

        self.fft_points_array = np.copy(sampled_fft_point_array)
        if output:
            return self.fft_points_array

    def plot_fft(self, fft_array=None, save=False):
        """
        plots three dimensional fft frequency spectrum of passed array.
        :param fft_array: complex nd-array to be plotted. if None class fft array is used
        :param save: conditional to save plot as html
        :return:
        """
        if fft_array is None:
            fft_array = self.fft_points_array
        sample_length = self.unaltered_point_array.shape[0]
        sample_range = np.linspace(1, sample_length, fft_array.shape[1])
        fft_array = np.transpose(fft_array)
        fft_array = np.copy(np.transpose(fft_array[:int(fft_array.shape[0] / 2)]))
        fig = make_subplots(rows=3, cols=1,
                            subplot_titles=("X-Fourier Transform", "Y-Fourier Transform", "Z-Fourier Transform"),
                            shared_xaxes=True)
        fig.add_trace(go.Scatter(x=sample_range, y=np.real(fft_array[0]),
                                 mode='lines',
                                 name='x-transform real'),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=sample_range, y=np.imag(fft_array[0]),
                                 mode='lines',
                                 name='x-transform imaginary'),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=sample_range, y=np.real(fft_array[1]),
                                 mode='lines',
                                 name='y-transform real'),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=sample_range, y=np.imag(fft_array[1]),
                                 mode='lines',
                                 name='y-transform imaginary'),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=sample_range, y=np.real(fft_array[2]),
                                 mode='lines',
                                 name='z-transform real'),
                      row=3, col=1)
        fig.add_trace(go.Scatter(x=sample_range, y=np.imag(fft_array[2]),
                                 mode='lines',
                                 name='z-transform imaginary'),
                      row=3, col=1)
        fig.update_yaxes(type='linear')
        fig.show()
        if save:
            fig.write_html('fft_sample_figure.html')

    def reconstruct_space(self, output=False):
        """
        Constructs nd-array of points using fft array. Point space matches shape of original points.
        :param output: conditional to keep method static or return output.
        :return:
        """

    def r_squared(self, perfect_array=None, fft_sampled_array=None):
        if perfect_array is None:
            perfect_array = self.unaltered_point_array
        if fft_sampled_array is None:
            fft_sampled_array = self.fft_points_array
        sampled_array = sp.fft.ifftn(fft_sampled_array)
        pass
