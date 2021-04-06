import datahandling
import fouriergenerator
import modelgenerator
import open3d as o3d

gm = fouriergenerator.GeometryManipulation(50000)

gm.make_coil(store=True)

gm.point_array = datahandling.fetch_coil_points()
gm.main_spiral = datahandling.fetch_coil_points(filename="docs/main_coil_points.json")

pc_array = datahandling.fetch_coil_points()

pc = modelgenerator.PointCloud(pc_array)

normals = gm.generate_normals_from_source(normalise=False)

pc.pcd.normals = o3d.utility.Vector3dVector(gm.generate_normals_from_source(normalise=False))

# pc.down_sample(vox_size=0.05)

pc.show_pcd(show_normals=True)

# msh = modelgenerator.Mesh(pcd=pc.pcd)

# msh.poisson_mesh(run_checks=False, depth=8, compute_normals=True)

# msh.smooth_taubin(iterations=10, compute_normals=True)

# o3d.io.write_triangle_mesh("docs/winding_coil.stl", msh.mesh)

# msh.show_mesh(wireframe=False)
