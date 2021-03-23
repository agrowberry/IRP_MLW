import datahandling
import fouriergenerator
import modelgenerator
import numpy as np
import plotly.graph_objects as go
import scipy as sp
import open3d as o3d

gm = fouriergenerator.GeometryManipulation(500000)

# gm.make_coil(store=True)

# datahandling.store_coil_points(gm.main_spiral, filename='docs/main_coil_points.json')

gm.point_array = datahandling.fetch_coil_points()
gm.main_spiral = datahandling.fetch_coil_points(filename='docs/main_coil_points.json')

pc_array = datahandling.fetch_coil_points()

pc = modelgenerator.PointCloud(pc_array)

pc.pcd.normals = o3d.utility.Vector3dVector(gm.generate_normals_from_source(normalise=False))

# pc.down_sample(vox_size=0.1)

msh = modelgenerator.Mesh(pcd=pc.pcd)

msh.poisson_mesh()

msh.mesh.compute_triangle_normals(normalized=True)

datahandling.write_mesh_stl(msh.mesh)



