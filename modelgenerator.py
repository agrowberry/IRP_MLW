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
        self.pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

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
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=avg_dist, max_nn=30))
        pass

    def show_pcd(self, pcd=None):
        if pcd is None:
            pcd = self.pcd
        o3d.visualization.draw_geometries([pcd], mesh_show_wireframe=True, point_show_normal=True)


class Mesh:
    def __init__(self, pcd=None, mesh=None):
        """
        :param pcd:  o3d.PointCloud with normals to form mesh object.
        """
        self.unmeshed_pcd = pcd
        self.mesh = mesh
        if self.mesh is None and self.unmeshed_pcd is None:
            raise Exception('either a pointcloud or mesh object need to be passed to initialise the class...')


    def bpa_mesh(self, pcd=None):
        if pcd is None:
            pcd = self.unmeshed_pcd
        print('initialising Mesh object with %s points' % 'hello')
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 1 * avg_dist
        self.mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,
                                                                                    o3d.utility.DoubleVector([radius,
                                                                                                              radius * 5]))

    def poisson_mesh(self, pcd=None):
        if pcd is None:
            pcd = self.unmeshed_pcd
        print('run Poisson surface reconstruction')
        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=8)
        print(mesh)
        self.mesh = mesh
        densities = np.asarray(densities)
        density_colors = plt.get_cmap('plasma')(
            (densities - densities.min()) / (densities.max() - densities.min()))
        density_colors = density_colors[:, :3]
        self.mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)

    def show_mesh(self, mesh=None):
        if mesh is None:
            mesh = self.mesh
        o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)


