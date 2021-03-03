import datahandling
import fouriergenerator


gm = fouriergenerator.GeometryManipulation()
fm = fouriergenerator.FourierManipulation()


gm.make_coil(50000, plot=True)

gm.fig.write_html('coil_figure.html')










