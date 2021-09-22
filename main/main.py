import numpy as np
import cupy as cp
import grid as g
import variables as var
import elliptic as ell
import plotter as my_plt
import fluxes as fx
import time as timer

# elements and order
elements, order = [16, 32], 8

# set up grid
lows = np.array([-np.pi, -4.0])
highs = np.array([np.pi, 4.0])
grid = g.PhaseSpace(lows=lows, highs=highs, elements=elements, order=order)

# build distribution
# test_scalar = var.SpaceScalar(resolution=elements[0])
# test_scalar.arr_nodal = cp.sin(grid.x.device_arr)
# test_scalar.fourier_transform()
initial_distribution = var.Distribution(resolutions=elements, order=order)
initial_distribution.initialize(grid=grid)
initial_distribution.fourier_transform()
initial_distribution.inverse_fourier_transform()

test_distribution = var.Distribution(resolutions=elements, order=order)
test_distribution.initialize(grid=grid)
test_distribution.fourier_transform()
test_distribution.inverse_fourier_transform()

diff_distribution = var.Distribution(resolutions=elements, order=order)

# test elliptic solver
elliptic = ell.Elliptic(resolution=elements[0])
elliptic.poisson_solve(distribution=test_distribution, grid=grid)

print(np.amax(elliptic.field.arr_nodal))

# test spectral flux reconstruction
flux = fx.DGFlux(resolutions=elements, order=order)
# flux.compute_flux(distribution=test_distribution, elliptic=elliptic)
flux.semi_discrete_rhs(distribution=test_distribution, elliptic=elliptic, grid=grid)

# A time-stepper (put in its own class!)
t0 = timer.time()
time = 0
dt = 1.0e-3
plotter = my_plt.Plotter(grid=grid)
plotter.distribution_contourf(distribution=test_distribution, plot_spectrum=True)
plotter.show()

for i in range(5):
    for j in range(1000):
        time += dt
        flux.semi_discrete_rhs(distribution=test_distribution, elliptic=elliptic, grid=grid)
        test_distribution.arr += dt * flux.output.arr
    print('Took step, time is {:0.3e}'.format(time))
    # timer.sleep(5)
    # For plotting:
    diff_distribution.arr = test_distribution.arr - initial_distribution.arr
    diff_distribution.inverse_fourier_transform()
    # zero and electric field
    test_distribution.zero_moment.inverse_fourier_transform()
    elliptic.field.inverse_fourier_transform()

    # Look at it:

    plotter.distribution_contourf(distribution=test_distribution, plot_spectrum=True)
    plotter.distribution_contourf(distribution=diff_distribution, plot_spectrum=False)
    # plotter.spatial_scalar_plot(scalar=test_distribution.zero_moment, y_axis='Zero moment')
    # plotter.spatial_scalar_plot(scalar=elliptic.field, y_axis='Electric Field', spectrum=False)
    plotter.show()
print('Done, it took {:0.3e}'.format(timer.time() - t0))


elliptic.field.inverse_fourier_transform()
print(np.amax(elliptic.field.arr_nodal))

diff_distribution.arr = test_distribution.arr - initial_distribution.arr

# zero and electric field
test_distribution.zero_moment.inverse_fourier_transform()
elliptic.field.inverse_fourier_transform()

plotter = my_plt.Plotter(grid=grid)
plotter.distribution_contourf(distribution=test_distribution)
plotter.distribution_contourf(distribution=diff_distribution)
plotter.spatial_scalar_plot(scalar=test_distribution.zero_moment, y_axis='Zero moment')
plotter.spatial_scalar_plot(scalar=elliptic.field, y_axis='Electric Field')
plotter.show()

# plotter.spatial_scalar_plot(scalar=test_scalar, y_axis='test')
# plotter.distribution_contourf(distribution=flux.flux)
# plotter.distribution_contourf(distribution=flux.output)
