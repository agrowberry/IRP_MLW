import scipy as sp
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import datahandling
from plotly.subplots import make_subplots
import open3d as o3d

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
        self.num_turns = None
        self.profile_ppt = None
        self.path_ppt = None
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
            (axis_length / 2 - coil_radius),
            # - (axis_length - 2 * coil_radius) / n_height,
            (-axis_length / 2 + coil_radius),
            # + (axis_length - 2 * coil_radius) / n_height,
            n_height
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
            (-axis_length / 2 + coil_radius),
            # + (axis_length - 2 * coil_radius) / n_height,
            (axis_length / 2 - coil_radius),
            # - (axis_length - 2 * coil_radius) / n_height,
            n_height
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
        num_per_turn,
        output=False,
        output_geometries=False,
    ):
        """
        produces square-helical 3d point array of n points.
        :param width:
        :param height:
        :param length:
        :param num_turns:
        :param num_per_turn:
        :param radius:
        :param output:
        :param output_geometries:
        :return:
        """
        # declarations and piecewise function describe coil, arbitrary function could be used.
        perimeter = (
            2 * (height - 2 * radius) + 2 * (width - 2 * radius) + 2 * np.pi * radius
        )
        # calculate number of points per section of rectangle to make even point spacing.
        n_w = int(round((width - 2 * radius) * (num_per_turn / perimeter)))
        n_h = int(round((height - 2 * radius) * (num_per_turn / perimeter)))
        n_r = int(round((2 * np.pi * radius) * (num_per_turn / perimeter)))

        # create dictionary for optional output
        geometry_dict = {"n width": n_w, "n height": n_h, "n radius": n_r}

        # make x and y coordinate arrays for one turn
        x = self.piecewise(height, radius, n_h, n_w, n_r, num_per_turn)
        y = self.piecewise(width, radius, n_w, n_h, n_r, num_per_turn, transpose=True)

        # form 2D array of coordinates for one turn
        vector_list = [list(np.copy(x)), list(np.copy(y)), ""]
        # extend array for number of turns in coil
        for i in range(num_turns - 1):
            vector_list[0].extend(x)
            vector_list[1].extend(y)
            datahandling.progressbar(i, num_turns - 1, message="building coil")
        # add z-axis component
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

    def map_points(self, unmapped_profile_array, normal_array, trans_array, output=False):
        """
        maps loops of unmapped profile array to path array, sweeps profile around paths
        k = n // ppt
        :param unmapped_profile_array: (nx3) profile array - unmapped
        :type unmapped_profile_array: ndarray
        :param normal_array: (kx3) gradient of path array - normal of unmapped plane
        :type normal_array: ndarray
        :param trans_array: (kx3) path array
        :type trans_array: ndarray
        :param output: kwarg conditional for making method static.
        :type output: bool
        :return: mapped_array: n-dimensional array of mapped points.
        """
        mapped_list = []
        k = int(self.path_ppt * self.num_turns)
        profile_matrix = np.reshape(unmapped_profile_array, (k, self.profile_ppt, 3))

        def angle(p):
            angle_rad = np.angle(p[0] + p[1] * 1j, deg=False)
            return angle_rad

        for index, path_point in enumerate(trans_array):
            normal_vector = normal_array[index]
            unmapped_profile_loop = profile_matrix[index]
            rot_angle_y = 1 * (np.pi / 2 - angle(
                np.array(
                    [
                        (normal_vector[0] ** 2 + normal_vector[1] ** 2) ** 0.5,
                        normal_vector[2],
                    ]
                )
            ))
            rot_angle_z = -1 * angle(np.array([normal_vector[0], normal_vector[1]]))
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
            mapped_matrix = np.copy(
                np.matmul(unmapped_profile_loop, np.matmul(r_y, r_z)) + path_point
            )
            mapped_list.append(mapped_matrix)

            datahandling.progressbar(
                index,
                len(trans_array),
                message="mapping profile points to coil path",
                exit_message="finished, coil built...",
            )
        if output:
            return np.concatenate(mapped_list)
        else:
            self.point_array = np.transpose(np.concatenate(mapped_list))

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
            index = cross.count
            if index + 1 < vector_array.shape[1]:
                lat_vector = v_array[index + 1] - vector
            else:
                lat_vector = vector - v_array[index - 1]
            if index + self.profile_ppt < vector_array.shape[1]:
                long_vector = v_array[index + self.profile_ppt] - vector
            else:
                long_vector = vector - v_array[index - self.profile_ppt]
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
            geom_dict = datahandling.read_dict('docs/dimensions.json')
        if n is None:
            n = self.n
        # find coil dimensions inputted dict.
        # path dimensions
        num_turns = int(geom_dict["num_of_turns"])
        major_width = geom_dict["core_major_axis"] + geom_dict["outer_spacing"]
        major_height = geom_dict["core_minor_axis"] + geom_dict["outer_spacing"]
        major_length = geom_dict["core_length"]
        major_radius = geom_dict["core_radius"]
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

        # check that n will decompose into 2*num_turns, if not, fix.
        a = n % (num_turns * 2)
        while n % (num_turns * 2) != 0:
            n += 1
        self.n = n

        profile_perimeter = (
            2 * (coil_height - 2 * coil_radius)
            + 2 * (coil_width - 2 * coil_radius)
            + 2 * np.pi * coil_radius
        )

        path_perimeter = (
            2 * major_height + 2 * major_width
            - 4 * major_radius
            + 2 * np.pi * major_radius
        )

        profile_ppt = int(((n * profile_perimeter)/(num_turns*path_perimeter))**0.5)

        if (n / num_turns) % profile_ppt == 0:
            indivisible = False
        else:
            indivisible = True
            distance = 1
        while indivisible:
            above = profile_ppt + distance
            below = profile_ppt - distance
            if (n / num_turns) % above == 0:
                profile_ppt = above
                indivisible = False
            if (n / num_turns) % below == 0:
                profile_ppt = below
                indivisible = False
            distance += 1

        path_ppt = n // (num_turns * profile_ppt)

        self.num_turns = num_turns
        self.profile_ppt = profile_ppt
        self.path_ppt = path_ppt

        print('points: %s \nnumber of turns: %s \npath ppt: %s\nprofile ppt: %s' %
              (self.n, self.num_turns, self.path_ppt, self.profile_ppt))

        # makes an array describing path shape.
        self.loop_size_dict = self.make_helix(
            major_width,
            major_height,
            major_length,
            num_turns,
            major_radius,
            path_ppt,
            output_geometries=True,
        )
        # find gradient array to be mapped to.
        self.grad(np.transpose(self.main_spiral))
        # make coil profile, mapped to path.
        # find best possible ratio of points per turn to number of turns of path array.
        self.coil_spiral = self.make_helix(
            coil_width, coil_height, 0, path_ppt*num_turns, coil_radius, profile_ppt, output=True
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
        profile_ppt=None,
        path_ppt=None,
        profile_array=None,
        path_array=None,
        loop_size_dict=None,
    ):
        """
        # breaks down coil profile and path components into storable dict for use in FEMM analysis.
        :param num_turns: number of coil turns
        :param profile_ppt: points per profile turn
        :param profile_array: unmapped profile array
        :param path_array: path_array
        :param loop_size_dict: dictionary containing allocation of turn points.
        :return:
        """
        # assign attributes to method variables if None.
        geom_dict = datahandling.read_dict("docs/dimensions.json")
        if num_turns is None:
            num_turns = geom_dict["num_of_turns"]
        if profile_ppt is None:
            profile_ppt = self.profile_ppt
        if path_ppt is None:
            path_ppt = self.path_ppt
        if profile_array is None:
            profile_array = self.coil_spiral
            equal = profile_array == np.zeros((3, self.n))
            if equal.all():
                profile_array = datahandling.fetch_coil_points('docs/coil_array_points.json')
        if path_array is None:
            path_array = np.transpose(self.main_spiral)
        if loop_size_dict is None:
            loop_size_dict = self.loop_size_dict
            if loop_size_dict is None:
                loop_size_dict = datahandling.read_dict("docs/loop_size_dict.json")

        # calc. coil parameters, for splitting up coil into useful profiles.
        loops = path_ppt * num_turns
        cross_sections = 2 * num_turns + 1
        loops_per_half_turn = loops / (num_turns * 2)
        print(
            "loops: %s \ncross-sections: %s \nloops per half turn: %s"
            % (loops, cross_sections, loops_per_half_turn)
        )
        # make dict with coil parameters
        header_dict = {
            "coil shape": geom_dict,
            "profile_ppt": profile_ppt,
            "path_ppt": path_ppt,
            "cross sections": cross_sections,
            "lpht": loops_per_half_turn,
            "loop size": loop_size_dict,
        }
        cross_sections_dict = {}
        profile_array = np.transpose(profile_array)
        # split coil into individual profile loops for 3D matrix.
        profile_matrix = np.reshape(profile_array, (int(loops), int(profile_ppt), 3))
        for i in range(cross_sections):
            index = int(i * loops_per_half_turn)
            if i == cross_sections - 1:
                cross_sections_dict[i] = (tuple(path_array[-1]), profile_matrix[-1])
            else:
                cross_sections_dict[i] = (tuple(path_array[index]), profile_matrix[index])

            datahandling.progressbar(i, cross_sections, 'breaking down sections')
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
        self.profile_ppt = info_dict["profile_ppt"]
        self.lpht = info_dict["lpht"]
        self.loop_size = info_dict["loop size"]
        self.cross_sections = info_dict["cross sections"]
        self.reconstructed_array = None

    def reconstruct_coil(self, output=True):
        # for each turn create a lofted matrix of profile shapes. loft for the turning section of the path.
        n_h = self.loop_size["n height"]

        if n_h % 2 == 0:
            start_point = n_h // 2
            end_point = n_h // 2
        else:
            start_point = (n_h - 1) // 2 + 1
            end_point = (n_h - 1) // 2

        # find n gradients for j loops between start and end loops.
        def _find_grad(start_vector, end_vector, size):
            shift_vector = (end_vector - start_vector) / (size + 2)
            return shift_vector

        find_grad = np.vectorize(_find_grad)

        # will move nth point in jth loop by j*grad_i, for loft.
        def _shift(start, grad, posn):
            vector = start + (posn + 1) * grad
            return vector

        shift = np.vectorize(_shift)

        # for jth point in jth loop will create linear loft using start profile and gradient to loft.
        # j: 0 -> n  | j: 0 -> k
        def loft(gradient, k, start):
            lofted_loop_list = []
            for i in range(k):
                lofted_loop = shift(start, gradient, k)
                lofted_loop_list.append(lofted_loop)
            return np.array(lofted_loop_list)

        lofting_points = (
            int(self.lpht - (start_point + end_point))
        )

        turn_profile_list = []
        for turn in range(self.cross_sections - 1):
            # start and end profiles are (ppt x 3) arrays
            # k = lpht
            start_profile = self.coil_loop_dict[turn][1]
            end_profile = self.coil_loop_dict[turn + 1][1]
            profile_sections_list = [np.tile(start_profile, (start_point, 1, 1))]
            gradient_array = find_grad(start_profile, end_profile, lofting_points)
            profile_sections_list.append(
                loft(gradient_array, lofting_points, start_profile)
            )
            profile_sections_list.append(np.tile(end_profile, (end_point, 1, 1)))
            turn_profile = np.concatenate(profile_sections_list)
            turn_profile_list.append(turn_profile)
            datahandling.progressbar(turn, self.cross_sections - 1, "lofting sections")
        lofted_profile_matrix = np.concatenate(turn_profile_list)
        lofted_profile_array = lofted_profile_matrix.reshape(
            -1, lofted_profile_matrix.shape[-1]
        )

        sub_gm = GeometryManipulation(lofted_profile_array.shape[0])
        mapping_normal_array = sub_gm.grad(self.path_array, output=True)
        sub_gm.path_ppt = int(self.lpht * 2)
        sub_gm.num_turns = self.path_array.shape[0] // sub_gm.path_ppt
        sub_gm.profile_ppt = int(self.profile_ppt)
        # self.pcd = o3d.geometry.PointCloud()
        # self.pcd.points = o3d.utility.Vector3dVector(lofted_profile_array)
        # o3d.visualization.draw_geometries([self.pcd])
        mapped_lofted_surface_array = sub_gm.map_points(
            lofted_profile_array, mapping_normal_array, self.path_array, output=True
        )
        if output:
            return mapped_lofted_surface_array
        else:
            self.reconstructed_array = mapped_lofted_surface_array


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
