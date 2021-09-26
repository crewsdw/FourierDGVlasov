import numpy as np
# import cupy as np
import grid as g
import variables as var
import elliptic as ell
import plotter as my_plt
import fluxes as fx
import time as timer
import scipy.integrate as spint
import timestep as ts
from copy import deepcopy

# elements and order
elements, order = [8, 20], 8

# set up grid
lows = np.array([-5*np.pi, -8])
highs = np.array([5*np.pi, 8])
grid = g.PhaseSpace(lows=lows, highs=highs, elements=elements, order=order)

# build distribution
# test_scalar = var.SpaceScalar(resolution=elements[0])
# test_scalar.arr_nodal = np.sin(grid.x.device_arr)
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
# flux = fx.DGFlux(resolutions=elements, order=order)
# flux.compute_flux(distribution=test_distribution, elliptic=elliptic)
# flux.semi_discrete_rhs(distribution=test_distribution, elliptic=elliptic, grid=grid)

# A time-stepper (put in its own class!)
t0 = timer.time()
time = 0
dt = 1.0e-3
step = 1.0e-1
dt_max = 1.0 / (np.amax(grid.x.wavenumbers) * np.amax(grid.v.arr))
print('Max dt is {:0.3e}'.format(dt_max))
plotter = my_plt.Plotter(grid=grid)
plotter.distribution_contourf(distribution=test_distribution, plot_spectrum=True)
# plotter.spatial_scalar_plot(scalar=elliptic.field, y_axis='Electric Field', spectrum=True)
plotter.show()

stepper = ts.Stepper(dt=dt, step=step, resolutions=elements, order=order, steps=50)
final_distribution = stepper.main_loop(distribution=test_distribution, elliptic=elliptic,
                                       grid=grid, plotter=plotter, plot=False)

# for i in range(300):
#     # for j in range(1000):
#     #     time += dt
#     #     flux.semi_discrete_rhs(distribution=test_distribution, elliptic=elliptic, grid=grid)
#     #     test_distribution.arr += dt * flux.output.arr
#     next_time = time + step
#     sol = spint.solve_ivp(fun=flux.semi_discrete_rhs, t_span=[time, next_time],
#                           y0=test_distribution.arr.flatten(), method='RK23', first_step=dt,
#                           vectorized=False, args=(test_distribution, elliptic, grid))
#     print(sol.keys)
#     test_distribution.arr = sol[0].reshape(test_distribution.arr.shape)
#     time += step
#     print('Took step, time is {:0.3e}'.format(time))
# timer.sleep(5)
# For plotting:
# diff_distribution.arr = deepcopy(test_distribution.arr - initial_distribution.arr)
# diff_distribution.inverse_fourier_transform()
# # zero and electric field
# test_distribution.zero_moment.inverse_fourier_transform()
# elliptic.field.inverse_fourier_transform()
#
# # Look at it:
#
# plotter.distribution_contourf(distribution=test_distribution, plot_spectrum=False)
# # plotter.distribution_contourf(distribution=diff_distribution, plot_spectrum=False)
# # plotter.spatial_scalar_plot(scalar=test_distribution.zero_moment, y_axis='Zero moment')
# # plotter.spatial_scalar_plot(scalar=elliptic.field, y_axis='Electric Field', spectrum=False)
# plotter.show()
print('Done, it took {:0.3e}'.format(timer.time() - t0))

elliptic.field.inverse_fourier_transform()
print(np.amax(elliptic.field.arr_nodal))
diff_distribution.arr = final_distribution.arr - initial_distribution.arr

# zero and electric field
test_distribution.zero_moment.inverse_fourier_transform()
elliptic.field.inverse_fourier_transform()

plotter = my_plt.Plotter(grid=grid)
plotter.distribution_contourf(distribution=test_distribution)
plotter.distribution_contourf(distribution=diff_distribution)
plotter.spatial_scalar_plot(scalar=test_distribution.zero_moment, y_axis='Zero moment')
plotter.spatial_scalar_plot(scalar=elliptic.field, y_axis='Electric Field')
plotter.time_series_plot(time=stepper.time_array, series=stepper.field_energy,
                         y_axis='Electric energy', log=True, give_rate=True)
plotter.time_series_plot(time=stepper.time_array, series=stepper.thermal_energy,
                         y_axis='Thermal energy', log=False)
plotter.time_series_plot(time=stepper.time_array, series=stepper.density_array,
                         y_axis='Total density', log=False)
plotter.time_series_plot(time=stepper.time_array, series=stepper.field_energy + stepper.thermal_energy,
                         y_axis='Total energy', log=False)
plotter.show()

# plotter.spatial_scalar_plot(scalar=test_scalar, y_axis='test')
# plotter.distribution_contourf(distribution=flux.flux)
# plotter.distribution_contourf(distribution=flux.output)
