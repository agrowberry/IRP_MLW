import open3d as o3d
import numpy as np
import datahandling
import pandas as pd
import OCC.Core as occ
import matplotlib.pyplot as plt


class PointCloud:
    def __init__(self, points_array, normal_array=None):
        self.pcd_array = points_array
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(self.pcd_array)
        if normal_array is None:
            self.pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
        else:
            self.pcd.normals = o3d.utility.Vector3dVector(normal_array)

    def down_sample(self, vox_size, full_pcd=None, output=False):
        """
        returns a down sampled o3d.PointCloud object
        :param vox_size: spacing of resultant voxels.
        :param full_pcd: specify an external pcd to be downsampled
        :param output: conditional to make method static
        :return down_pcd: optional returned down sampled pcd array.
        """
        if full_pcd is None:
            full_pcd = self.pcd
        down_pcd = full_pcd.voxel_down_sample(voxel_size=vox_size)
        if output:
            return down_pcd
        else:
            self.pcd = down_pcd

    def find_normals(self, pcd=None):
        if pcd is None:
            pcd = self.pcd
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=avg_dist, max_nn=30
            )
        )
        pass

    def show_pcd(self, pcd=None, show_normals=False):
        if pcd is None:
            pcd = self.pcd
        o3d.visualization.draw_geometries(
            [pcd], mesh_show_wireframe=True, point_show_normal=show_normals
        )

    @staticmethod
    def generate_normals_from_source(
            ppt, point_array, normalise=True, output=True
    ):
        """
        Creates an array of normal vectors (optionally normalised) of the mapped profile surface.
        Normals generated from taking cross-product of two vectors along surface.
        Method is mirrored from fouriergenerator.py GeometryManipulation class.
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

        def cross(vector, vector_array):
            v_array = np.transpose(vector_array)
            index = cross.count
            if index + 1 < vector_array.shape[1]:
                lat_vector = v_array[index + 1] - vector
            else:
                lat_vector = vector - v_array[index - 1]
            if index + ppt < vector_array.shape[1]:
                long_vector = v_array[index + ppt] - vector
            else:
                long_vector = vector - v_array[index - ppt]
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
        if output:
            return normal_array


class Mesh:
    def __init__(self, pcd=None, mesh=None):
        """
        :param pcd:  o3d.PointCloud with normals to form mesh object.
        """
        self.unmeshed_pcd = pcd
        self.mesh = mesh
        if self.mesh is None and self.unmeshed_pcd is None:
            raise Exception(
                "either a PointCloud or TriangleMesh object need to be passed to initialise the class..."
            )

    def run_checks(self, mesh=None):
        """
        runs checks for mesh, watertight checks for if the shape is closed and forms a volume.
        intersecting checks for any self-intersections on the shape, will return a list of the intersecting triangles.
        :param mesh: TriangleMesh object to be checked. if None the class .mesh attribute
        :type mesh: open3d TriangleMesh object.
        :return:
        """
        if mesh is None:
            mesh = self.mesh
        yes_or_no = ["is", "isn't"]
        print("running checks on Mesh:...\n")

        print("checking for self_intersections:...\n")
        with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug
        ) as cm:
            intersecting = mesh.is_self_intersecting()
        print("mesh %s self-intersecting\n" % (yes_or_no[int(intersecting)]))
        if intersecting:
            intersecting_triangles = mesh.get_self_intersecting_triangles()
            print("following triangles are intersecting:\n")
            print(np.asarray(intersecting_triangles))

        print("checking for closed shape:...\n")
        watertight = mesh.is_watertight()
        print("mesh %s closed\n" % (yes_or_no[int(watertight)]))
        while not watertight:
            yes_answer = ["y", "Y", "Yes", "YES", "yes"]
            no_answer = ["n", "N", "no", "NO", "No"]
            answer = input("mesh is not closed. Attempt to close? (y/n)")
            if answer in yes_answer:
                mesh.merge_close_vertices()
            if answer in no_answer:
                break
            watertight = mesh.is_watertight()
            if watertight:
                print("mesh is closed")

    def bpa_mesh(self, pcd=None, radius_scalar=1, output=False, run_checks=False):
        """
        produces TriangleMesh object using ball-pivoting-algorithm from passed PointCloud object
        :param pcd: PointCloud array to be meshed if none class pcd object is used.
        :type pcd: PointCloud
        :param radius_scalar: scalar of average distances between points in PointCloud to use in BPA
        :type radius_scalar: int
        :param output: conditional for making method static.
        :type output: bool
        :return mesh: optionally returned TriangleMesh object
        """
        if pcd is None:
            pcd = self.unmeshed_pcd
        print("initialising Mesh object using:\n %s" % (str(pcd)))
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = radius_scalar * avg_dist
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, radii=o3d.utility.DoubleVector([0.01*radius, 0.1*radius, 0.5*radius, radius, 1.5*radius])
        )
        if output:
            return mesh
        else:
            self.mesh = mesh
        if run_checks:
            self.run_checks()

    def poisson_mesh(
        self, pcd=None, depth=8, output=False, compute_normals=True, run_checks=False
    ):
        """
        produces TriangleMesh object using poisson meshing algorithm from passed PointCloud object
        :param pcd: PointCloud array to be meshed if none class pcd object is used.
        :type pcd: PointCloud
        :param depth: number of passes the poisson algorithm performs on object.
        :type depth: int
        :param output: conditional for making method static.
        :type output: bool
        :param compute_normals: conditional for computing face normals for the mesh
        :type compute_normals: bool
        :param run_checks: conditional for running mesh geometry checks
        :type output: bool
        :return mesh: optional returned TriangleMesh object
        """
        if pcd is None:
            pcd = self.unmeshed_pcd
        print("running Poisson surface reconstruction with %s passes" % depth)
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth
        )
        print(mesh)
        densities = np.asarray(densities)
        density_colors = plt.get_cmap("plasma")(
            (densities - densities.min()) / (densities.max() - densities.min())
        )
        density_colors = density_colors[:, :3]
        mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
        mesh.remove_duplicated_vertices()
        if compute_normals:
            mesh.compute_triangle_normals(normalized=True)
        if output:
            return mesh
        else:
            self.mesh = mesh
        if run_checks:
            self.run_checks()

    def piecewise_poisson_mesh(
        self, points, normals, chunks, depth = 8, output = False, compute_normals = True, run_checks = False
    ):
        """
        produces TriangleMesh object using poisson meshing algorithm from passed PointCloud object
        :param pcd: PointCloud array to be meshed if none class pcd object is used.
        :type pcd: PointCloud
        :param depth: number of passes the poisson algorithm performs on object.
        :type depth: int
        :param output: conditional for making method static.
        :type output: bool
        :param compute_normals: conditional for computing face normals for the mesh
        :type compute_normals: bool
        :param run_checks: conditional for running mesh geometry checks
        :type output: bool
        :return mesh: optional returned TriangleMesh object
        """
        chunk_size = int(points.shape[0] / chunks)
        meshed_chunks = o3d.geometry.TriangleMesh()
        for chunk in range(chunks):
            start = chunk_size*(chunk)
            stop = chunk_size*(chunk + 1)
            chunk_points = points[start: stop]
            chunk_normals = normals[start: stop]
            chunk_pcd = o3d.geometry.PointCloud()
            chunk_pcd.points = o3d.utility.Vector3dVector(chunk_points)
            chunk_pcd.normals = o3d.utility.Vector3dVector(chunk_normals)
            print("running Poisson surface reconstruction on chunk %s" % chunk)
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                chunk_pcd, depth=depth
            )
            meshed_chunks += mesh
        mesh = meshed_chunks
        mesh.remove_duplicated_vertices()
        mesh.paint_uniform_color(np.array([214, 109, 109]))
        if compute_normals:
            mesh.compute_triangle_normals(normalized=True)
        if output:
            return mesh
        else:
            self.mesh = mesh
        self.mesh = meshed_chunks


    def smooth_laplacian(
        self, mesh=None, iterations=1, compute_normals=True, output=False
    ):
        if mesh is None:
            mesh = self.mesh
        print("running laplacian smoothing on mesh with %s passes" % iterations)
        mesh = o3d.geometry.TriangleMesh.filter_smooth_laplacian(
            mesh, number_of_iterations=iterations
        )
        if compute_normals:
            mesh.compute_triangle_normals(normalized=True)
        if output:
            return mesh
        else:
            self.mesh = mesh

    def smooth_taubin(
        self, mesh=None, iterations=1, compute_normals=True, output=False
    ):
        if mesh is None:
            mesh = self.mesh
        print("running taubin smoothing on mesh with %s passes" % (iterations))
        mesh = o3d.geometry.TriangleMesh.filter_smooth_taubin(
            mesh, number_of_iterations=iterations
        )
        if compute_normals:
            mesh.compute_triangle_normals(normalized=True)
        if output:
            return mesh
        else:
            self.mesh = mesh

    def close_mesh(
        self,
        mesh=None,
        eps=0.05,
        store_original="docs/winding_coil_tmp.stl",
        output=False,
    ):
        if mesh is None:
            mesh = self.mesh
        print("closing mesh, merging vertices closer than %s" % eps)
        closed = mesh.is_watertight()
        if type(store_original) is str:
            o3d.io.write_stl_file(store_original, mesh)
        if closed:
            print("mesh is already closed. Skipping...")
            pass
        else:
            mesh = o3d.geometry.TriangleMesh.merge_close_vertices(mesh, eps=eps)
        if output:
            return mesh
        else:
            self.mesh = mesh

    def show_mesh(self, mesh=None, wireframe=True):
        if mesh is None:
            mesh = self.mesh
        o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=wireframe)
