import scipy as sp
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import datahandling
from plotly.subplots import make_subplots


class GeometryManipulation:
    """
    class object representing real-space coil.
    """

    def __init__(self, num=None):
        self.n = num
        self.fig = go.Figure()
        self.geom_dict = {}
        self.point_array = np.zeros((3, self.n))
        self.local_spiral = np.zeros((3, self.n))
        self.main_spiral = np.zeros((3, self.n))
        self.grad_array = np.zeros((3, self.n))
        self.coil_spiral = np.zeros((3, self.n))
        self.normal_array = np.zeros((3, self.n))
        self.no_of_profile_turns = None
        self.ppt = None

    def make_helix(self, width, height, length, num_turns, radius, output=False):
        """
        produces square-helical 3d point array of n points.
        :param width:
        :param height:
        :param length:
        :param num_turns:
        :param radius:
        :param output:
        :return:
        """
        num_per_turn = int(round(self.n / num_turns))
        perimeter = (
            2 * (height - 2 * radius) + 2 * (width - 2 * radius) + 2 * np.pi * radius
        )
        n_w = int(round((width - 2 * radius) * (num_per_turn / perimeter)))
        n_h = int(round((height - 2 * radius) * (num_per_turn / perimeter)))
        n_r = int(round((2 * np.pi * radius) * (num_per_turn / perimeter)))

        def piecewise(coil_height, coil_radius, n_height, n_width):
            corner_array = coil_radius * np.sin(np.linspace(0, 2 * np.pi, n_r))
            returned_array = np.full(n_width, coil_height / 2)
            returned_array = np.append(
                returned_array,
                corner_array[len(corner_array) // 4 : len(corner_array) // 2]
                + coil_height / 2
                - coil_radius,
            )
            returned_array = np.append(
                returned_array,
                np.linspace(
                    coil_height / 2 - coil_radius,
                    -coil_height / 2 + coil_radius,
                    n_height,
                ),
            )
            returned_array = np.append(
                returned_array,
                corner_array[2 * len(corner_array) // 4 : 3 * len(corner_array) // 4]
                - coil_height / 2
                + coil_radius,
            )
            returned_array = np.append(
                returned_array, np.full(n_width, -coil_height / 2)
            )
            returned_array = np.append(
                returned_array,
                corner_array[3 * len(corner_array) // 4 :]
                - coil_height / 2
                + coil_radius,
            )
            returned_array = np.append(
                returned_array,
                np.linspace(
                    -coil_height / 2 + coil_radius,
                    coil_height / 2 - coil_radius,
                    n_height,
                ),
            )
            returned_array = np.append(
                returned_array,
                corner_array[: len(corner_array) // 4] + coil_height / 2 - coil_radius,
            )
            if len(returned_array) < num_per_turn:
                returned_array = np.append(
                    returned_array,
                    np.full(num_per_turn - len(returned_array), returned_array[-1]),
                )
            if len(returned_array) > num_per_turn:
                returned_array = returned_array[:num_per_turn]

            return returned_array

        x = piecewise(height, radius, n_h, n_w)
        y = piecewise(width, radius, n_w, n_h)
        y = np.append(y[n_h + n_r // 4 :], y[: n_h + n_r // 4])

        vector_list = [list(np.copy(x)), list(np.copy(y)), ""]
        for i in range(num_turns - 1):
            vector_list[0].extend(x)
            vector_list[1].extend(y)
            datahandling.progressbar(i, num_turns - 1, message="building coil")
        vector_list[2] = np.linspace(0, length, len(vector_list[0]))
        helix_array = np.array(
            [np.array(vector_list[0]), np.array(vector_list[1]), vector_list[2]]
        )
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
        df = pd.DataFrame(
            {"x": vector_array[0], "y": vector_array[1], "z": vector_array[2]}
        )
        grad_df = df.diff()
        grad_df.iloc[0] = grad_df.iloc[1]
        datahandling.workingmessage.count = 0
        for index, row in grad_df.iterrows():
            if np.abs(row["x"]) < 0.0001 and np.abs(row["y"]) < 0.0001:
                datahandling.workingmessage("removing zero points")
                grad_df.at[index, "x"] = np.mean(
                    np.array(
                        [
                            grad_df.iloc[int(index + 1), 0],
                            grad_df.iloc[int(index - 1), 0],
                        ]
                    )
                )
                grad_df.at[index, "y"] = np.mean(
                    np.array([grad_df.iloc[index + 1, 1], grad_df.iloc[index - 1, 1]])
                )
        print("gradient array built", end="\n")
        grad_array = grad_df.to_numpy()
        if normalise:
            grad_array = (
                grad_array
                / np.apply_along_axis(np.linalg.norm, 0, np.transpose(grad_array))[
                    :, None
                ]
            )
        if output:
            return grad_array
        else:
            self.grad_array = np.copy(grad_array)

    def map_points(self, unmapped_local_array, normal_array, trans_array, output=False):
        """
        will map point from global origin [0, 0, 0] to translated and rotated points
        :param unmapped_local_array: n-dimensional array of points to be mapped.
        :type unmapped_local_array: ndarray
        :param normal_array: n-dimensional array of normals to rotation point.
        :type normal_array: ndarray
        :param trans_array: n-dimensional array of translated origins.
        :type trans_array: ndarray
        :param output: kwarg conditional for making method static.
        :type output: bool
        :return: mapped_array: n-dimensional array of mapped points.
        """
        mapped_list = []

        def angle(p):
            angle_rad = np.angle(p[0] + p[1] * 1j, deg=False)
            return angle_rad

        for index, unmapped_vector in enumerate(unmapped_local_array):
            normal_vector = normal_array[index]
            unmapped_vector = unmapped_local_array[index]
            trans_vector = trans_array[index]
            rot_angle_y = np.pi / 2 - angle(
                np.array(
                    [
                        (normal_vector[0] ** 2 + normal_vector[1] ** 2) ** 0.5,
                        normal_vector[2],
                    ]
                )
            )
            rot_angle_z = angle(np.array([normal_vector[0], normal_vector[1]]))
            r_y = np.copy(
                np.array(
                    [
                        np.array([np.cos(rot_angle_y), 0, np.sin(rot_angle_y)]),
                        np.array([0, 1, 0]),
                        np.array([-np.sin(rot_angle_y), 0, np.cos(rot_angle_y)]),
                    ]
                )
            )
            r_z = np.copy(
                np.array(
                    [
                        np.array([np.cos(rot_angle_z), -np.sin(rot_angle_z), 0]),
                        np.array([np.sin(rot_angle_z), np.cos(rot_angle_z), 0]),
                        np.array([0, 0, 1]),
                    ]
                )
            )
            mapped_vector = np.copy(
                np.matmul(r_z, np.matmul(r_y, unmapped_vector)) + trans_vector
            )
            mapped_list.append(mapped_vector)
            datahandling.progressbar(
                index,
                len(unmapped_local_array),
                message="mapping profile points to coil path",
                exit_message="finished, coil built...",
            )
        if output:
            return np.array(mapped_list)
        else:
            self.point_array = np.transpose(np.array(mapped_list))

    def make_coil(self, n=None, geom_dict=None, plot=False, store=True):
        """
        main script calling functions to make a 3D coil of designated size.
        :param store: conditional to store returned array in .json file.
        :type store: bool
        :param plot: conditional to plot returned array.
        :type plot: bool
        :param: geom_dict, dictionary in std_input_dict format with geometry params.
        :type geom_dict: dict
        :param n: no. of points in returned array.
        :type n: int
        :return: array of points describing square coil around rectangular core.
        """
        if geom_dict is None:
            geom_dict = datahandling.start_up()
        if n is None:
            n = self.n
        #     find coil dimensions inputted dict.
        num_turns = int(geom_dict["num_of_turns"])
        coil_height = (
            geom_dict["core_length"] - (num_turns - 1) * geom_dict["spacing"]
        ) / num_turns
        coil_width = geom_dict["outer_spacing"] - 2 * geom_dict["spacing"]
        coil_radius = geom_dict["coil_radius_percentage"] * (
            min(coil_width, coil_height)
        )
        major_width = (
            geom_dict["core_major_axis"] + geom_dict["spacing"] + coil_width / 2
        )
        major_height = (
            geom_dict["core_minor_axis"] + geom_dict["spacing"] + coil_height / 2
        )
        major_length = geom_dict["core_length"]
        major_radius = geom_dict["core_radius"]

        # makes coil path describing main shape.
        self.make_helix(
            major_width, major_height, major_length, num_turns, major_radius
        )
        # find gradient array to be mapped to.
        self.grad(np.transpose(self.main_spiral))
        # make coil profile, mapped to path.
        # find best possible ratio of points per turn to number of turns of path array.
        # find perimeter of coil profile
        perimeter = (
            2 * (coil_height - 2 * coil_radius)
            + 2 * (coil_width - 2 * coil_radius)
            + 2 * np.pi * coil_radius
        )
        # find vector from one point to another
        path_spacing_vector = (
            np.transpose(self.main_spiral)[1] - np.transpose(self.main_spiral)[0]
        )
        # find distance of two points
        path_spacing = (
            path_spacing_vector[0] ** 2
            + path_spacing_vector[1] ** 2
            + path_spacing_vector[2] ** 2
        ) ** 0.5
        # solves for positive real soln for equal spacing between profile points in lateral and longitudinal direction.
        num_per_turn = (
            ((4 * perimeter ** 2 + path_spacing ** 2) ** 0.5) / path_spacing + 1
        ) ** 0.5 / (2 ** 0.5)
        # number of profile turns = number of total points / number of points per turn.
        num_of_profile_turns = int(round(n / num_per_turn))
        # if the number of points in the array is not divisible by the number of profile turns then the nearest factor is found and returned.
        if n % num_of_profile_turns == 0:
            indivisible = False
        else:
            indivisible = True
            distance = 1
        while indivisible:
            above = num_of_profile_turns + distance
            below = num_of_profile_turns - distance
            if n % above == 0:
                num_of_profile_turns = above
                indivisible = False
            if n % below == 0:
                num_of_profile_turns = below
                indivisible = False
            distance += 1
        self.no_of_profile_turns = num_of_profile_turns
        self.ppt = n//num_of_profile_turns
        self.coil_spiral = self.make_helix(
            coil_width, coil_height, 0, num_of_profile_turns, coil_radius, output=True
        )
        # maps profile to path
        self.map_points(
            np.transpose(self.coil_spiral),
            self.grad_array,
            np.transpose(self.main_spiral),
        )
        if plot:
            # plots outputted shape using plotly.
            self.plot_point_array(
                major_length,
                major_width,
                major_height,
                coil_width,
                coil_height,
                store="png",
            )
        if store:
            # stores mapped profile to .json file.
            datahandling.store_coil_points(self.point_array)
            # stores path to .json file.
            datahandling.store_coil_points(
                self.main_spiral, filename="docs/main_coil_points.json"
            )

    def generate_normals_from_source_deprecated(
        self, point_array=None, path_array=None, normalise=True, output=True
    ):
        """
        creates an array of normal vectors (optionally normalised) of the mapped profile surface.
        normals generated as direction from the point-n in the path to point-n in the mapped profile.
        :param point_array: mapped profile 3-dimensional array of N-points. forms 'surface'.
        :type point_array: ndarray
        :param path_array: path of coil to make normal vector from surface.
        :type path_array: ndarray
        :param normalise: kwarg conditional to normalise vector output.
        :type normalise: bool
        :param output: kwarg conditional to make method static and return normal array.
        :type output: bool
        :return normal_array: optionally returned ndarray of normals.
        """
        if point_array is None:
            point_array = np.transpose(self.point_array)
        if path_array is None:
            path_array = np.transpose(self.main_spiral)
        self.normal_array = point_array - path_array
        if normalise:

            def normalise(vector):
                magnitude = np.linalg.norm(vector)
                if magnitude == 0:
                    return vector
                else:
                    return vector / magnitude

            self.normal_array = np.apply_along_axis(normalise, 0, self.normal_array)
        if output:
            return self.normal_array

    def generate_normals_from_source(
        self, point_array=None, normalise=True, output=True
    ):
        """
        Creates an array of normal vectors (optionally normalised) of the mapped profile surface.
        Normals generated from taking cross-product of two vectors along surface. 
        :param point_array: mapped profile 3-dimensional array of N-points. forms 'surface'.
        :type point_array: ndarray
        :param path_array: path of coil to make normal vector from surface.
        :type path_array: ndarray
        :param normalise: kwarg conditional to normalise vector output.
        :type normalise: bool
        :param output: kwarg conditional to make method static and return normal array.
        :type output: bool
        :return:
        """
        if point_array is None:
            point_array = self.point_array
        
        def cross(vector, vector_array):
            v_array = np.transpose(vector_array)
            index = np.where(v_array==vector)[0]
            if index + 1 >= vector_array.shape[1]:
                lat_vector = v_array[index + 1] - vector
            else:
                lat_vector = vector - v_array[index - 1]
            if index + self.ppt >= vector_array.shape[1]:
                long_vector = v_array[index + self.ppt] - vector
            else:
                long_vector = vector - v_array[index - self.ppt]
            return np.cross(lat_vector, long_vector)

        normal_array = np.apply_along_axis(cross, 0, point_array, point_array)


        if normalise:

            def normalise(vector):
                magnitude = np.linalg.norm(vector)
                if magnitude == 0:
                    return vector
                else:
                    return vector / magnitude

            normal_array = np.apply_along_axis(normalise, 0, normal_array)

        self.normal_array = normal_array
        if output:
            return self.normal_array

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
        self.fig.add_trace(
            go.Scatter3d(
                x=self.point_array[0],
                y=self.point_array[1],
                z=self.point_array[2],
                mode="lines",
                name="Coil Surface",
            )
        )
        self.fig.add_trace(
            go.Scatter3d(
                x=self.main_spiral[0],
                y=self.main_spiral[1],
                z=self.main_spiral[2],
                name="Coil Path",
            )
        )

        max_main_dim = max(m_l, m_w, m_h)
        max_local_dim = max(c_w, c_h)
        axis_lims = [
            -(max_main_dim / 2 + max_local_dim + 1),
            (max_main_dim / 2 + max_local_dim + 1),
        ]
        self.fig.update_layout(
            scene=dict(
                xaxis=dict(range=axis_lims),
                yaxis=dict(range=axis_lims),
                zaxis=dict(range=axis_lims),
            )
        )
        if store is not None:
            self.fig.write_image("point_array_scatter", format=store)

        self.fig.show()


class FourierManipulation:
    """
    class object representing frequency domain coil.
    """

    def __init__(self):
        self.fft_points_array = np.zeros((3, 1000))
        self.unaltered_point_array = np.transpose(datahandling.fetch_coil_points())
        self.reconstructed_point_array = np.zeros(self.unaltered_point_array.shape)

    @staticmethod
    def next_power2(n):
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
            deletable_indices.append(
                int(
                    round(((i + 0.5) / down_sample_num) * len(self.fft_points_array))
                    + offset
                )
            )
        # creates list of indices to sample the unaltered_point_array.
        sampled_indices = list(
            set(range(len(self.fft_points_array))) - set(deletable_indices)
        )
        # maps the list of indices to be sampled to an nd array of sampled points.
        sampled_fft_point_array = np.transpose(
            [*map(self.fft_points_array.__getitem__, sampled_indices)]
        )

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
        fft_array = np.copy(np.transpose(fft_array[: int(fft_array.shape[0] / 2)]))
        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=(
                "X-Fourier Transform",
                "Y-Fourier Transform",
                "Z-Fourier Transform",
            ),
            shared_xaxes=True,
        )
        fig.add_trace(
            go.Scatter(
                x=sample_range,
                y=np.real(fft_array[0]),
                mode="lines",
                name="x-transform real",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=sample_range,
                y=np.imag(fft_array[0]),
                mode="lines",
                name="x-transform imaginary",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=sample_range,
                y=np.real(fft_array[1]),
                mode="lines",
                name="y-transform real",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=sample_range,
                y=np.imag(fft_array[1]),
                mode="lines",
                name="y-transform imaginary",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=sample_range,
                y=np.real(fft_array[2]),
                mode="lines",
                name="z-transform real",
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=sample_range,
                y=np.imag(fft_array[2]),
                mode="lines",
                name="z-transform imaginary",
            ),
            row=3,
            col=1,
        )
        fig.update_yaxes(type="linear")
        fig.show()
        if save:
            fig.write_html("fft_sample_figure.html")

    def reconstruct_space(self, output=False):
        """
        Constructs nd-array of points using fft array. Point space matches shape of original points.
        :param output: conditional to keep method static or return output.
        :return:
        """
        pass

    def r_squared(self, perfect_array=None, fft_sampled_array=None):
        if perfect_array is None:
            perfect_array = self.unaltered_point_array
        if fft_sampled_array is None:
            fft_sampled_array = self.fft_points_array
        sampled_array = sp.fft.ifftn(fft_sampled_array)
        pass
