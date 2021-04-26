import datahandling
import numpy as np
import fouriergenerator
import modelgenerator
import open3d as o3d
import femm
import plotly.graph_objects as go

gm = fouriergenerator.GeometryManipulation(1_000_000)

# gm.make_coil(plot=False, store=True, generate_normals=True)
#
# header_info = {'ppt': gm.ppt,
#                "core_length": 10.0,
#                "core_minor_axis": 10.0,
#                "core_major_axis": 20.0,
#                "core_radius": 1.0,
#                "coil_radius_percentage": 0.25,
#                "num_of_turns": 8,
#                "outer_spacing": 5.0,
#                "spacing": 0.1,
#                }
#
# datahandling.store_complete_info(header_info, gm.point_array, gm.surface_normals, 'docs/coil_geometry_info.json')

header, pc_array, pc_normals = datahandling.fetch_complete_info('docs/coil_geometry_info.json')

coil_dict = gm.breakdown_coil(ppt=header['ppt'], profile_array=pc_array)

cc = fouriergenerator.CompactCoil(coil_dict)

cc.reconstruct_coil()

x_sec_dict = coil_dict['x-sec']

fig = go.Figure()

# x_list = []
# for k, v in x_sec_dict.items():
#     array = np.transpose(v)
#     fig.add_trace(go.Scatter3d(x=array[0], y=array[1], z=array[2], name=k))
#     x_list.append(array)
#
# x_array = np.concatenate(tuple(x_list))

# pc = modelgenerator.PointCloud(pc_array)
#
# pc.pcd.normals = o3d.utility.Vector3dVector(np.transpose(pc_normals))
#
# pc.show_pcd(show_normals=True)

# msh = modelgenerator.Mesh(pcd=pc.pcd)
#
# msh.poisson_mesh(run_checks=False, depth=10, compute_normals=True)
#
# o3d.io.write_triangle_mesh("docs/winding_coil.stl", msh.mesh)
#
# msh.show_mesh(wireframe=False)
