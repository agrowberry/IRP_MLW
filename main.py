import datahandling
import numpy as np
import fouriergenerator
import modelgenerator
import open3d as o3d
import surfacegenerator
import femm
import femmhandling
import plotly.graph_objects as go

gm = surfacegenerator.GeometryManipulation(1_000_000)

simple_coil, size_dict = gm.make_helix(30, 15, 0, 1, 3, 60, output=True, output_geometries=True)

datahandling.store_coil_points(simple_coil, filename="C:/Users/rowbe/IRP_MLW/simple_coil.json")











# gm.make_coil(plot=False, store=False, generate_normals=False)

# coil_dict = gm.breakdown_coil()
#
# pp = femmhandling.Preprocessor()
#
# pp.build_coil(coil_dict['x-sec'])

# datahandling.store_dict(coil_dict, filename='docs/compact_coil.json')

# cc = surfacegenerator.CompactCoil(coil_dict)
# 
# cc.reconstruct_coil(output=False)
# 
# pc = modelgenerator.PointCloud(cc.reconstructed_array)
# 
# pc.show_pcd(show_normals=False)

# msh = modelgenerator.Mesh(pcd=pc.pcd)
#
# msh.poisson_mesh(run_checks=False, depth=10, compute_normals=True)
#
# o3d.io.write_triangle_mesh("docs/winding_coil.stl", msh.mesh)


# msh.show_mesh(wireframe=False)
