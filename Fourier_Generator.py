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
        self.point_array = np.array(np.zeros((3, 1000)))
        self.local_spiral = np.array(np.zeros(3, 100))
        self.main_spiral = np.array(np.zeros(3, 100))
        self.grad_array = np.array(np.zeros(3, 100))

    def make_rectangle(self, width, height, length, n, output=True, return_nd_array=False):
        """
        returns rectangular helix of specified dimensions in array.
        :param width:
        :param height:
        :param length:
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

        magnitude = lambda p1, p2: ((p1[0] - p2[0])**2 + (p1[1] - p2[2])**2)**0.5

        for i in range(1, len(corners)):
            corner = corners[i]
            last_corner = corners[i - 1]
            x_points = np.linspace(last_corner[0], corner[0], int(magnitude(corner, last_corner) * n / perimeter))
            x_list.append(x_points)
            y_points = np.linspace(last_corner[1], corner[1], int(magnitude(corner, last_corner) * n / perimeter))
            y_list.append(y_points)

        x = np.concatenate(x_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        z = np.linspace(0, length, y.shape[0])

        if output:
            if return_nd_array:
                return np.array(x, y, z)
            else:
                return x, y, z
        else:
            self.local_spiral = np.array(x, y, z)

    def make_helix(self, single_spiral, no_of_turns, output=False):

        helix_list = [[], [], []]
        length = single_spiral[2][-1]

        for i in range(no_of_turns):
            nth_spiral = np.copy(single_spiral)
            nth_spiral[2] = single_spiral[2] + i * length
            for j in range(3):
                helix_list[j].append(nth_spiral[j])

            helix_array = np.array([np.concatenate(helix_list[0]),
                                    np.concatenate(helix_list[1]),
                                    np.concatenate(helix_list[2])])

        if output:
            return helix_array
        else:
            self.main_spiral = np.copy(helix_array)

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

        return grad_array

    def map_points(self, unmapped_local_array, normal_array, trans_array):
        """
        will map point from global origin [0, 0, 0] to translated and rotated points
        :param unmapped_local_array: n-dimensional array of points to be mapped.
        :param normal_array: n-dimensional array of normals to rotation point.
        :param trans_array: n-dimensional array of translated origins.
        :return: mapped_array: n-dimensional array of mapped points.
        """

        origin_normal = np.array([0, 1, 0])
        points_df = pd.DataFrame(normal_array, columns=['normal'])
        points_df['unmapped'] = unmapped_local_array
        points_df['translation'] = trans_array
        points_df['mapped'] = np.array(np.zeros(unmapped_local_array.shape()))

        for index, row in points_df.iterrows():
            normal_vector = row['normal']
            c = np.dot(origin_normal, normal_vector)
            axis = np.unitcross(origin_normal, normal_vector)
            s = np.sqrt(1 - c**2)
            C = 1 - c
            x = axis[0]
            y = axis[1]
            z = axis[2]
            rot_matrix = np.array([[x*x*C+c, x*y*C-z*s, x*z*C+y*s],
                                   [y*x*C+z*s, y*y*C+c, y*z*C-x*s],
                                   [z*x*C-y*s, z*y*C+x*s, z*z*C+c]])

            points_df['mapped'][index] = np.dot(rot_matrix, row['unmapped']) + row['translation']

        return pd.array(points_df['mapped'])

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


fg = FourierManipulation()

main_spiral = fg.make_rectangle()




