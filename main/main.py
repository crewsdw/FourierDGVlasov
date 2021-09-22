import numpy as np
import cupy as cp
import grid as g
import variables as var
import elliptic as ell
import plotter as my_plt

# elements and order
elements, order = [32, 32], 3

# set up grid
lows = np.array([-np.pi, -4.0])
highs = np.array([np.pi, 4.0])
grid = g.PhaseSpace(lows=lows, highs=highs, elements=elements, order=order)

# build distribution
# test_scalar = var.SpaceScalar(resolution=elements[0])
# test_scalar.arr_nodal = cp.sin(grid.x.device_arr)
# test_scalar.fourier_transform()
test_distribution = var.Distribution(resolutions=elements, order=order)
test_distribution.initialize(grid=grid)
test_distribution.fourier_transform()

# test elliptic solver
elliptic = ell.Elliptic(resolution=elements[0])
elliptic.poisson_solve(distribution=test_distribution, grid=grid)

plotter = my_plt.Plotter(grid=grid)
# plotter.spatial_scalar_plot(scalar=test_scalar, y_axis='test')
plotter.distribution_contourf(distribution=test_distribution)
plotter.spatial_scalar_plot(scalar=test_distribution.zero_moment, y_axis='Zero moment')
plotter.spatial_scalar_plot(scalar=elliptic.field, y_axis='Electric Field')
plotter.show()
