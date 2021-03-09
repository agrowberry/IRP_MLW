import datahandling
import fouriergenerator
import numpy as np
import plotly.graph_objects as go
import scipy as sp

gm = fouriergenerator.GeometryManipulation()
fm = fouriergenerator.FourierManipulation()

gm.make_coil(50000, plot=True)

