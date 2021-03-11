# IRP_MLW
Individual Research Project | Machine Learning Winding

This project provides several tools for programmatically designing, evaluating and optimising single-slot electrical windings.

---
 ## Fourier Series Module (fouriergenerator.py)

Provides a set of tools for constructing basic coil winding geometries. With further tools for parameterising the construction. 

The two main objects in the module are .point_array and .fft_points_array:
- .point_array exists in the real space and is a numpy 3-dimensional array of points describing the winding coil.

- .fft_points_array is the frequency domain counterpart to .point_array. It represents the three fast fourier transformed coefficients