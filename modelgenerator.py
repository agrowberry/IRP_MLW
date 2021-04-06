import open3d as o3d
import numpy as np
import datahandling
import pandas as pd
import OCC.Core as occ
import matplotlib.pyplot as plt


class PointCloud:
    def __init__(self, points_array):
        self.pcd_array = points_array
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(np.transpose(self.pcd_array))
        self.pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )

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
            pcd, radii=o3d.utility.DoubleVector([radius, radius * 5])
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
        produces TriangleMesh object using poisson smoothing algorithm from passed PointCloud object
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
        if compute_normals:
            mesh.compute_triangle_normals(normalized=True)
        if output:
            return mesh
        else:
            self.mesh = mesh
        if run_checks:
            self.run_checks()

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
        print("closing mesh, merging vertices closer than %s" % (eps))
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
