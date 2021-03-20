import datahandling
import fouriergenerator
import numpy as np
import plotly.graph_objects as go
import scipy as sp
import open3d as o3d

gm = fouriergenerator.GeometryManipulation(500000)
fm = fouriergenerator.FourierManipulation()

gm.make_coil(plot=True)

pc_array = datahandling.fetch_coil_points()





