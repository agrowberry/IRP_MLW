import datahandling
import fouriergenerator
import modelgenerator
import numpy as np
import plotly.graph_objects as go
import scipy as sp
import open3d as o3d

gm = fouriergenerator.GeometryManipulation(1000000)

# gm.make_coil(store=True)
#
# datahandling.store_coil_points(gm.main_spiral, filename='docs/main_coil_points.json')

gm.point_array = datahandling.fetch_coil_points()
gm.main_spiral = datahandling.fetch_coil_points(filename='docs/main_coil_points.json')

pc_array = datahandling.fetch_coil_points()

pc = modelgenerator.PointCloud(pc_array)

pc.pcd.normals = o3d.utility.Vector3dVector(gm.generate_normals_from_source(normalise=False))

pc.down_sample(vox_size=0.05)

# pc.show_pcd()

msh = modelgenerator.Mesh(pcd=pc.pcd)

msh.poisson_mesh(run_checks=False, depth=8, compute_normals=True)

msh.mesh.compute_triangle_normals(normalized=True)

msh.smooth_taubin(iterations=10)

msh.mesh.compute_triangle_normals(normalized=True)

o3d.io.write_triangle_mesh('docs/winding_coil.stl', msh.mesh)

msh.show_mesh(wireframe=False)
