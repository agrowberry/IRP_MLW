import datahandling
import fouriergenerator


gm = fouriergenerator.GeometryManipulation()
fm = fouriergenerator.FourierManipulation()

<<<<<<< Updated upstream
gm.make_coil(50000, plot=True)

# pc_array = np.transpose(datahandling.fetch_coil_points())


# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(pc_array)
# pcd.estimate_normals(
#     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
# )
# pcd = pcd.voxel_down_sample(voxel_size=0.05)
# # distances = pcd.compute_nearest_neighbor_distance()
# # avg_dist = np.mean(distances)
# # radius = 1 * avg_dist
# #
# # bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius,
# #                                                                                                          radius * 2]))
#
# o3d.visualization.draw_geometries([pcd])
#
#
#
=======

gm.make_coil(50000, plot=True)

gm.fig.write_html('coil_figure.html', include_plotlyjs='cdn')










>>>>>>> Stashed changes
