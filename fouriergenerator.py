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
        self.surface_normals = None
        self.loop_size_dict = None

    @staticmethod
    def piecewise(
        axis_length, coil_radius, n_height, n_width, n_r, num_per_turn, transpose=False
    ):
        """
        static method to produce single dimensional component of a rounded rectangle.
        :param axis_length:
        :param coil_radius:
        :param n_height:
        :param n_width:
        :param n_r:
        :return:
        """
        corner_array = coil_radius * np.sin(np.linspace(0, 2 * np.pi, n_r))
        sections = ["", "", "", "", "", "", "", ""]
        # RH Straight
        sections[0] = np.full(n_width, axis_length / 2)
        # Top RH Corner
        sections[1] = (
            corner_array[round(len(corner_array) / 4) : round(len(corner_array) / 2)]
            + axis_length / 2
            - coil_radius
        )
        # Top Straight
        sections[2] = np.linspace(
            (axis_length / 2 - coil_radius)
            - (axis_length - 2 * coil_radius) / n_height,
            (-axis_length / 2 + coil_radius)
            + (axis_length - 2 * coil_radius) / n_height,
            n_height,
        )
        # Top LH Corner
        sections[3] = (
            corner_array[
                round(len(corner_array) / 2) : round((3 * len(corner_array)) / 4)
            ]
            - axis_length / 2
            + coil_radius
        )
        # LH Straight
        sections[4] = np.full(n_width, -axis_length / 2)
        # Bottom LH Corner
        sections[5] = (
            corner_array[round((3 * len(corner_array)) / 4) :]
            - axis_length / 2
            + coil_radius
        )
        # Bottom Straight
        sections[6] = np.linspace(
            (-axis_length / 2 + coil_radius)
            + (axis_length - 2 * coil_radius) / n_height,
            (axis_length / 2 - coil_radius)
            - (axis_length - 2 * coil_radius) / n_height,
            n_height,
        )
        # Bottom RH Corner
        sections[7] = (
            corner_array[: round(len(corner_array) / 4)] + axis_length / 2 - coil_radius
        )

        # shifts order of sections to provide alternate axis.
        if transpose:
            shifted_sections = sections[6:]
            shifted_sections += sections[:6]
            # returned_array = np.concatenate(tuple(shifted_sections))
            finished_list = shifted_sections
        else:
            # returned_array = np.concatenate(tuple(sections))
            finished_list = sections
        # section 0 needs to be split to make midpoint of RH Straight the start/end point
        # take first half of section 0 and append to back of
        split_section = finished_list[0]
        if len(split_section) % 2 == 0:
            finished_list[0] = split_section[len(split_section) // 2 :]
            finished_list.append(split_section[: len(split_section) // 2])
        else:
            finished_list[0] = split_section[int((len(split_section) - 1) / 2 + 1) :]
            finished_list.append(split_section[: int((len(split_section) - 1) / 2 + 1)])

        returned_array = np.concatenate(tuple(finished_list))

        if len(returned_array) < num_per_turn:
            returned_array = np.append(
                returned_array,
                np.full(num_per_turn - len(returned_array), returned_array[-1]),
            )
        if len(returned_array) > num_per_turn:
            returned_array = returned_array[:num_per_turn]

        return returned_array

    def make_helix(
        self,
        width,
        height,
        length,
        num_turns,
        radius,
        output=False,
        output_geometries=False,
    ):
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
        # declarations and piecewise function describe coil, arbitrary function could be used.
        num_per_turn = int(round(self.n / num_turns))
        perimeter = (
            2 * (height - 2 * radius) + 2 * (width - 2 * radius) + 2 * np.pi * radius
        )
        n_w = int(round((width - 2 * radius) * (num_per_turn / perimeter)))
        n_h = int(round((height - 2 * radius) * (num_per_turn / perimeter)))
        n_r = int(round((2 * np.pi * radius) * (num_per_turn / perimeter)))

        geometry_dict = {"n width": n_w, "n_height": n_h, "n radius": n_r}

        x = self.piecewise(height, radius, n_h, n_w, n_r, num_per_turn)
        y = self.piecewise(width, radius, n_w, n_h, n_r, num_per_turn, transpose=True)

        vector_list = [list(np.copy(x)), list(np.copy(y)), ""]
        for i in range(num_turns - 1):
            vector_list[0].extend(x)
            vector_list[1].extend(y)
            datahandling.progressbar(i, num_turns - 1, message="building coil")
        vector_list[2] = np.linspace(0, length, len(vector_list[0]))
        helix_array = np.array(
            [np.array(vector_list[0]), np.array(vector_list[1]), vector_list[2]]
        )
        if output and output_geometries:
            return helix_array, geometry_dict
        elif not output and not output_geometries:
            self.main_spiral = np.copy(helix_array)
        elif not output and output_geometries:
            self.main_spiral = np.copy(helix_array)
            return geometry_dict
        if output and not output_geometries:
            return helix_array

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
        if type(self.ppt) is None:
            print("class has no coil instance, create coil before generating normals.")
            pass

        if point_array is None:
            point_array = self.point_array

        def cross(vector, vector_array):
            v_array = np.transpose(vector_array)
            index = cross.count
            if index + 1 < vector_array.shape[1]:
                lat_vector = v_array[index + 1] - vector
            else:
                lat_vector = vector - v_array[index - 1]
            if index + self.ppt < vector_array.shape[1]:
                long_vector = v_array[index + self.ppt] - vector
            else:
                long_vector = vector - v_array[index - self.ppt]
            cross.count += 1
            return np.cross(lat_vector, long_vector)

        cross.count = 0
        normal_array = np.apply_along_axis(cross, 0, point_array, point_array)

        if normalise:

            def normalise(vector):
                magnitude = np.linalg.norm(vector)
                if magnitude == 0:
                    return vector
                else:
                    return vector / magnitude

            normal_array = np.apply_along_axis(normalise, 0, normal_array)

        normal_array = np.transpose(normal_array)
        self.normal_array = normal_array
        if output:
            return self.normal_array

    def make_coil(
        self, n=None, geom_dict=None, generate_normals=True, plot=False, store=True
    ):
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
        # path dimensions
        num_turns = int(geom_dict["num_of_turns"])
        major_width = geom_dict["core_major_axis"] + geom_dict["outer_spacing"]
        major_height = geom_dict["core_minor_axis"] + geom_dict["outer_spacing"]
        major_length = geom_dict["core_length"]
        major_radius = geom_dict["core_radius"]
        # profile dimensions
        # coil_height = (
        #     geom_dict["core_length"] - (num_turns - 1) * geom_dict["spacing"]
        # ) / num_turns
        coil_height = (geom_dict["core_length"] / num_turns) * np.cos(
            np.arctan(
                (geom_dict["core_length"] / num_turns)
                / (2 * (major_width + major_height - major_radius))
            )
        ) - geom_dict["spacing"]
        coil_width = geom_dict["outer_spacing"] - 2 * geom_dict["spacing"]
        coil_radius = geom_dict["coil_radius_percentage"] * (
            min(coil_width, coil_height)
        )

        # check for collision condition in profile mapping (R < w/2)
        if major_radius < coil_width / 2:
            major_radius = 1.1 * coil_width / 2
            print("fixed bad path radius. new path radius: %s" % major_length)
            geom_dict["core_radius"] = major_radius

        # makes coil path describing main shape.
        self.loop_size_dict = self.make_helix(
            major_width,
            major_height,
            major_length,
            num_turns,
            major_radius,
            output_geometries=True,
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
        self.ppt = n // num_of_profile_turns
        self.coil_spiral = self.make_helix(
            coil_width, coil_height, 0, num_of_profile_turns, coil_radius, output=True
        )
        # maps profile to path
        self.map_points(
            np.transpose(self.coil_spiral),
            self.grad_array,
            np.transpose(self.main_spiral),
        )
        if generate_normals:
            self.surface_normals = self.generate_normals_from_source(normalise=True)
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
            datahandling.store_dict(geom_dict, "docs/dimensions.json")
            datahandling.store_dict(self.loop_size_dict, "docs/loop_size_dict.json")
            # stores mapped profile to .json file.
            datahandling.store_coil_points(self.point_array)
            # stores path to .json file.
            datahandling.store_coil_points(
                self.main_spiral, filename="docs/main_coil_points.json"
            )
            if type(self.surface_normals) is not None:
                datahandling.store_coil_points(
                    np.transpose(self.surface_normals),
                    filename="docs/main_coil_normals.json",
                )

    def breakdown_coil(
        self,
        num_turns=None,
        ppt=None,
        profile_array=None,
        path_array=None,
        loop_size_dict=None,
    ):
        # assign attributes to method variables if None.
        geom_dict = datahandling.read_dict("docs/dimensions.json")
        if num_turns is None:
            num_turns = geom_dict["num_of_turns"]
        if ppt is None:
            ppt = self.ppt
        if profile_array is None:
            profile_array = self.coil_spiral
        if path_array is None:
            path_array = self.main_spiral
        if loop_size_dict is None:
            loop_size_dict = self.loop_size_dict
            if loop_size_dict is None:
                loop_size_dict = datahandling.read_dict("docs/loop_size_dict.json")

        # calc. coil parameters, for splitting up coil into useful profiles.
        loops = profile_array.shape[1] / ppt
        cross_sections = 2 * num_turns + 1
        loops_per_half_turn = loops / (num_turns * 2)
        print(
            "loops: %s \ncross-sections: %s \nloops per half turn: %s"
            % (loops, cross_sections, loops_per_half_turn)
        )
        # make dict with coil parameters
        header_dict = {
            "coil shape": geom_dict,
            "ppt": ppt,
            "cross sections": cross_sections,
            "lpht": loops_per_half_turn,
            "loop size": loop_size_dict,
        }
        cross_sections_dict = {}
        profile_array = np.transpose(profile_array)
        # split coil into individual profile loops for 3D matrix.
        point_matrix = np.reshape(profile_array, (int(loops), int(ppt), 3))
        # take the profile loop at every half-turn for the compact dict.
        for i in range(cross_sections):
            index = int(i * loops_per_half_turn)
            if i == cross_sections - 1:
                cross_sections_dict[i] = point_matrix[-1]
            else:
                cross_sections_dict[i] = point_matrix[index]
        compact_coil_dict = {
            "header": header_dict,
            "x-sec": cross_sections_dict,
            "path": path_array,
        }
        return compact_coil_dict

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
        # self.fig.add_trace(
        #     go.Scatter3d(
        #         x=self.point_array[0],
        #         y=self.point_array[1],
        #         z=self.point_array[2],
        #         mode="lines",
        #         name="Coil Surface",
        #     )
        # )
        # self.fig.add_trace(
        #     go.Scatter3d(
        #         x=self.main_spiral[0],
        #         y=self.main_spiral[1],
        #         z=self.main_spiral[2],
        #         name="Coil Path",
        #     )
        # )

        self.fig.add_trace(
            go.Scatter3d(
                x=self.coil_spiral[0],
                y=self.coil_spiral[1],
                z=self.coil_spiral[2],
                mode="markers",
                name="Coil Profile",
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


class CompactCoil:
    def __init__(self, compact_coil_dict):
        info_dict = compact_coil_dict["header"]
        self.coil_loop_dict = compact_coil_dict["x-sec"]
        self.path_array = compact_coil_dict["path"]
        self.coil_path_dict = info_dict["coil shape"]
        self.ppt = info_dict["ppt"]
        self.lpht = info_dict["lpht"]
        self.loop_size = info_dict["loop size"]
        self.cross_sections = info_dict["cross sections"]

    def reconstruct_coil(self, output=True):
        # for each turn create a lofted matrix of profile shapes. loft for the turning section of the path.
        n_h = self.loop_size["n height"]
        n_w = self.loop_size['n width']
        n_r = self.loop_size['n radius']

        if n_h % 2 == 0:
            start_point = n_h // 2
            end_point = n_h // 2
        else:
            start_point = (n_h - 1) // 2 + 1
            end_point = (n_h - 1) // 2

        # fixes loft start and end points to be a set integer number of turns.
        if start_point % self.ppt == 0:
            divisible = True
        else:
            divisible = False
            distance = 1
        while not divisible:
            if (start_point + distance) % self.ppt == 0:
                divisible = True
                start_point += distance
            else:
                distance += 1

        if end_point % self.ppt == 0:
            divisible = True
        else:
            divisible = False
            distance = 1

        while not divisible:
            if (end_point + distance) % self.ppt == 0:
                divisible = True
                end_point += distance
            else:
                distance += 1

        lofting_points = (self.lpht*self.ppt) - (start_point + end_point)

        def _find_grad(start_vector, end_vector, size):
            shift_vector = (end_vector - start_vector)/(size + 2)
            return shift_vector
        find_grad = np.vectorize(_find_grad)

        def _shift()

        # for jth point in ith loop will create linear loft using start profile and gradient to loft.
        # j: 0 -> n         | i: 0 -> k
        def loft(start, gradient, k):
            for i in range(k):




        for turn in range(self.cross_sections):
            # start and end profiles are (ppt x 3) arrays
            # k = start_point + end_point + lofting_points

            start_profile = self.coil_loop_dict[turn]
            end_profile = self.coil_loop_dict[turn + 1]
            path_sections_list = [np.tile(start_profile, (start_point, 1, 1))]
            gradient_array = find_grad(start_profile, end_profile, lofting_points)

            print('oops')






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
