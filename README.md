# IRP_MLW
Individual Research Project | Machine Learning Winding

This project provides several tools for programmatically designing, evaluating and optimising single-slot electrical windings.

---
 ## Fourier Series Module (fouriergenerator.py)

Provides a set of tools for constructing basic coil winding geometries. With further tools for parameterising the construction. 

The two main objects in the module are .point_array and .fft_points_array:
- .point_array exists in the real space and is a numpy 3-dimensional array of points describing the winding coil.
([point array visualisation](https://agrowberry.github.io/IRP_MLW/coil_figure.html))
- .fft_points_array is the frequency domain counterpart to .point_array. It represents the three fast fourier transformed coefficients. The coefficients are stored as a 3-dimensional array of complex coefficients.
([fft array visualisation](https://agrowberry.github.io/IRP_MLW/fft_sample_figure.html))

The main aim of this module is to provide the user with a dimensionally reduced representation of the 