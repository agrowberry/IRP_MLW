import scipy as sp
import numpy as np
import plotly.graph_objects as go
import pandas as pd


class DataEntry(object):
    def __init__(self):
        self.std_inputs_dict = {'core_length': 10.0,
                                'core_minor_axis': 10.0,
                                'core_major_axis': 20.0,
                                'num_of_turns': 10.0,
                                'outer_spacing': 3.0,
                                'spacing': 0.5
                                }
        self.yes_answer = ['y', 'Y', 'Yes', 'YES', 'yes']
        self.no_answer = ['n', 'N', 'no', 'NO', 'No']

    def input_data(self, input_dict=False):
        if not input_dict:
            input_dict = self.std_inputs_dict
        else:
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

    def start_up(self):
        running = True
        while running:
            pass_manually = input('pass dimensions manually? (y/n): ')
            if pass_manually in self.yes_answer:
                input_dict = self.input_data(input_dict=True)
                running = False
                return input_dict
            elif pass_manually in self.no_answer:
                input_dict = self.std_inputs_dict
                # filename = input('enter filepath of coil dimensions: ')
                # input_dict = self.read_data(filename)
                running = False
                return input_dict
            else:
                pass


class FourierManipulation(object):
    def __init__(self):
        self.fig = go.Figure()
        self.geom_dict = {}
        self.point_array = np.zeros((3, 1000))
        self.local_spiral = np.zeros((3, 1000))
        self.main_spiral = np.zeros((3, 1000))
        self.grad_array = np.zeros((3, 1000))
        self.coil_sing_spiral = np.zeros((3, 1000))
        self.coil_spiral = np.zeros((3, 1000))

    def make_rectangle(self, width, height, length, n, output=False, return_nd_array=False):
        """
        returns rectangular helix of specified dimensions in array.
        :param width: width of rectangle
        :param height:height of rectangle
        :param length:length of coil revolution
        :param output: conditional dictating whether or not method is static and outputs array, or keeps attribute
         within class.
        :param return_nd_array: conditional for return format.
        :param n: no. of points in helix.
        :return: x, y, z || or 3d array of these variables.
        """
        corners = [(width / 2, 0),
                   (width / 2, height / 2),
                   (-width / 2, height / 2),
                   (-width / 2, -height / 2),
                   (width / 2, -height / 2),
                   (width / 2, 0)]
        x_list = []
        y_list = []
        perimeter = 2 * width + 2 * height
        magnitude = lambda p1, p2: ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
        for i in range(1, len(corners)):
            corner = corners[i]
            last_corner = corners[i - 1]
            x_points = np.linspace(last_corner[0],
                                   corner[0], int(round(magnitude(corner, last_corner) * n / perimeter)))
            x_list.append(x_points)
            y_points = np.linspace(last_corner[1],
                                   corner[1], int(round(magnitude(corner, last_corner) * n / perimeter)))
            y_list.append(y_points)
        x = np.concatenate(x_list, axis=0)
        y = np.concatenate(y_list, axis=0)

        # fix length discrepancy caused by rounding error.
        if x.shape[0] != n:
            diff_len = n - x.shape[0]
            x = np.append(x, np.full(int(diff_len), x[-1]))
            y = np.append(y, np.full(int(diff_len), y[-1]))

        z = np.linspace(0, length, y.shape[0])
        if output:
            if return_nd_array:
                return np.array([x, y, z])
            else:
                return x, y, z
        else:
            self.local_spiral = np.array([x, y, z])

    def make_helix(self, single_spiral, no_of_turns, output=False):
        helix_list = [[], [], []]
        helix_array = np.zeros(single_spiral.shape)
        length = single_spiral[2][-1]
        for i in range(no_of_turns):
            nth_spiral = np.copy(single_spiral)
            nth_spiral[2] = single_spiral[2] + i * length
            for j in range(3):
                helix_list[j].append(nth_spiral[j])
            helix_array = np.array([np.concatenate(helix_list[0]),
                                    np.concatenate(helix_list[1]),
                                    np.concatenate(helix_list[2])])
            print('building spiral ' + str(i) + ' of ' + str(no_of_turns) + ' : ' + str(100 * i / no_of_turns) + '%',
                  end='\r')
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
        :return: grad array: n-dimensional array of sequential gradients.
        """
        count = 1
        grad_array = []
        last_i = 0
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
        :return: mapped_array: n-dimensional array of mapped points.
        """
        origin_normal = np.array([0, 0, 1])
        mapped_list = []
        for i in range(unmapped_local_array.shape[0]):
            normal_vector = normal_array[i]
            unmapped_vector = unmapped_local_array[i]
            trans_vector = trans_array[i]
            c = np.dot(origin_normal, normal_vector)
            unnorm_axis = np.cross(origin_normal, normal_vector)
            det_unnorm_axis = (unnorm_axis[0]**2 + unnorm_axis[1]**2 + unnorm_axis[2]**2)**0.5
            axis = unnorm_axis * (1 / det_unnorm_axis)
            s = np.sqrt(1 - c ** 2)
            C = 1 - c
            x = axis[0]
            y = axis[1]
            z = axis[2]
            rot_matrix = np.array([[x*x*C+c, x*y*C-z*s, x*z*C+y*s],
                                   [y*x*C+z*s, y*y*C+c, y*z*C-x*s],
                                   [z*x*C-y*s, z*y*C+x*s, z*z*C+c]])
            mapped_vector = np.dot(rot_matrix, unmapped_vector) + trans_vector
            mapped_list.append(mapped_vector)
            print('mapping points ' + str(i) + ' of ' + str(unmapped_local_array.shape[0]), end='\r')
        if output:
            return np.array(mapped_list)
        else:
            self.point_array = np.transpose(np.array(mapped_list))

    def make_coil(self, N, geom_dict=True, plot=False):
        """
        :param: geom_dict, dictionary in std_input_dict format with geometry params.
        :param: N, no. of points in returned array.
        :return: array of points describing square coil around rectangular core.
        """
        de_local = DataEntry()
        if geom_dict:
            geom_dict = de_local.start_up()

        num_turns = int(geom_dict['num_of_turns'])
        coil_height = (geom_dict['core_length'] - (num_turns - 1)*geom_dict['spacing'])/num_turns
        coil_width = geom_dict['outer_spacing'] - 2*geom_dict['spacing']
        major_width = geom_dict['core_major_axis'] + geom_dict['spacing'] + coil_width/2
        major_height = geom_dict['core_minor_axis'] +geom_dict['spacing'] + coil_height/2
        major_length = geom_dict['core_length']/num_turns

        self.make_rectangle(major_width, major_height, major_length, int(round(N/num_turns)))
        self.make_helix(self.local_spiral, num_turns)
        self.grad(self.main_spiral)

        points_per_coil_spiral = 20
        self.coil_sing_spiral = self.make_rectangle(coil_width, coil_height, 0, points_per_coil_spiral,
                                                    output=True, return_nd_array=True)
        self.coil_spiral = self.make_helix(self.coil_sing_spiral, int(round(N/points_per_coil_spiral)), output=True)

        self.map_points(np.transpose(self.coil_spiral), np.transpose(self.grad_array), np.transpose(self.main_spiral))

        if plot:
            self.fig.add_trace(go.Scatter3d(x=self.point_array[0], y=self.point_array[1], z=self.point_array[2],
                                            mode='lines'))
            self.fig.add_trace(go.Scatter3d(x=self.main_spiral[0], y=self.main_spiral[1], z=self.main_spiral[2]))
            self.fig.add_trace(go.Scatter3d(x=self.coil_spiral[0], y=self.coil_spiral[1], z=self.coil_spiral[2]))
            self.fig.show()

    # returns array with fourier series coefficients.
    def make_fourier_series(self, geometry_dict, plot_result=True):
        val_list = []
        for val in geometry_dict:
            val_list.append(geometry_dict[val])


fg = FourierManipulation()

fg.make_coil(50000, plot=True)

