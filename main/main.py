import numpy as np
import grid as g
import variables as var
import elliptic as ell
import plotter as my_plt
import time as timer
import timestep as ts
import data

# Geometry and grid parameters
elements, order = [750, 25], 10
vt = 1
chi = 0.05
vb = 5
vtb = chi ** (1 / 3) * vb

# Grids
length = 2.0 * np.pi / 0.126  # 500  # 5000  # 1000
lows = np.array([-length / 2, -25 * vt])
highs = np.array([length / 2, 25 * vt])
grid = g.PhaseSpace(lows=lows, highs=highs, elements=elements, order=order)

# Build distribution
Distribution = var.Distribution(resolutions=elements, order=order, charge_mass=-1.0)
# Distribution.initialize_two_stream(grid=grid, vt1=1, vt2=1, u1=5, u2=-5)
# Distribution.initialize_random_two_stream(grid=grid, vt1=1, vt2=1, u1=5, u2=-5)
Distribution.initialize_eigenmode_two_stream(grid=grid, vt1=1, vt2=1, u1=5, u2=-5)
# Distribution.initialize_bump_on_tail(grid=grid, vt=vt, u=0, chi=chi, vb=vb, vtb=vtb)
Distribution.fourier_transform(), Distribution.inverse_fourier_transform()

# Set up elliptic problem
Elliptic = ell.Elliptic(resolution=elements[0])
Elliptic.poisson_solve_single_species(distribution=Distribution, grid=grid)

Plotter = my_plt.Plotter(grid=grid)
Plotter.distribution_contourf(distribution=Distribution, plot_spectrum=True,
                              remove_average=False, save='initial_condition')
# Plotter.spatial_scalar_plot(scalar=Distribution.zero_moment, y_axis='Zero moment electrons')
Plotter.spatial_scalar_plot(scalar=Elliptic.field, y_axis='Electric field', quadratic=True)
Plotter.show()

# Time integration class and stepping information
t0 = timer.time()
time = 0
dt = 1.0e-3  # 4.7e-4
step = 1.0e-3  # 4.7e-4
final_time = 14  # 31  # 101  # 172  # 151  # 100  # 100  # 150  # 50
steps = int(np.abs(final_time // step))
dt_max_translate = 1.0 / (np.amax(grid.x.wavenumbers) * np.amax(grid.v.arr)) / (2 * order + 1)
cutoff_velocity = 1.0 / (np.amax(grid.x.wavenumbers) * dt) / (2 * order + 1)
print('Max dt translation is {:0.3e}'.format(dt_max_translate))
print('Cutoff velocity at max wavenumber is {:0.3e}'.format(cutoff_velocity))

# Save data
# DataFile = data.Data(folder='..\\bot\\', filename='bot_5000LD_t' + str(final_time))
DataFile = data.Data(folder='..\\ts\\', filename='random_perturbation' + str(final_time))
DataFile.create_file(distribution=Distribution.arr_nodal.get(),
                     density=Distribution.zero_moment.arr_nodal.get(), field=Elliptic.field.arr_nodal.get())

# Set up stepper and execute main loop
Stepper = ts.StepperSingleSpecies(dt=dt, step=step, resolutions=elements, order=order,
                                  steps=steps, grid=grid, nu=0)
Stepper.main_loop_adams_bashforth(distribution=Distribution,
                                  elliptic=Elliptic, grid=grid, DataFile=DataFile)
Elliptic.field.inverse_fourier_transform()
print('Done, it took {:0.3e}'.format(timer.time() - t0))

# Final visualize
Plotter.distribution_contourf(distribution=Distribution, remove_average=False)

Plotter.spatial_scalar_plot(scalar=Distribution.zero_moment, y_axis='Density')
Plotter.spatial_scalar_plot(scalar=Elliptic.field, y_axis='Electric Field')

numpy_or_no = False
Plotter.time_series_plot(time_in=Stepper.time_array, series_in=Stepper.field_energy,
                         y_axis='Electric energy', log=True, give_rate=False, numpy=numpy_or_no)
Plotter.time_series_plot(time_in=Stepper.time_array, series_in=Stepper.thermal_energy-Stepper.thermal_energy[0],
                         y_axis='Thermal energy electrons', log=False, numpy=numpy_or_no)
Plotter.time_series_plot(time_in=Stepper.time_array, series_in=Stepper.density_array,
                         y_axis='Total density electrons', log=False, numpy=numpy_or_no)
Plotter.spatial_scalar_plot(scalar=Elliptic.field, y_axis='Field power spectral density', quadratic=True)
# plotter.animate_line_plot(saved_array=stepper.saved_density)
total_energy = Stepper.field_energy + Stepper.thermal_energy
Plotter.time_series_plot(time_in=Stepper.time_array, series_in=total_energy,
                         y_axis='Total energy', log=False, numpy=numpy_or_no)

# Save inventories
# print('Saving particle and energy inventories...')
# DataFile.save_inventories(total_energy=total_energy, total_density=Stepper.density_array)

# Look at plots
Plotter.show()

# plotter.spatial_scalar_plot(scalar=test_scalar, y_axis='test')
# plotter.distribution_contourf(distribution=flux.flux)
# plotter.distribution_contourf(distribution=flux.output)

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

# ####################################### Two species set-up
# # elements and order
# elements, order = [32, 100], 8
# mass_ratio = 1836  # m_p / m_e
# temp_ratio = 1  # T_e / T_p
# vt_e = 1
# vt_p = vt_e / np.sqrt(mass_ratio * temp_ratio)
#
# # set up grids
# wave_number = 0.5  # 0.05  # 0.1
# length = 2.0 * np.pi / wave_number
# lows_e = np.array([0, -12 * vt_e])
# highs_e = np.array([length, 12 * vt_e])
# grid_e = g.PhaseSpace(lows=lows_e, highs=highs_e, elements=elements, order=order)
#
# lows_p = np.array([0, -30 * vt_p])
# highs_p = np.array([length, 30 * vt_p])
# grid_p = g.PhaseSpace(lows=lows_p, highs=highs_p, elements=elements, order=order)
#
# print(grid_e.v.dx)
#
# # container
# grids = [grid_e, grid_p]
#
# # build distribution
# distribution_e = var.Distribution(resolutions=elements, order=order, charge_mass=-1.0)
# distribution_e.initialize(grid=grid_e, vt=vt_e, drift=0)  # 3
# distribution_e.fourier_transform(), distribution_e.inverse_fourier_transform()
#
# distribution_p = var.Distribution(resolutions=elements, order=order, charge_mass=+1.0 / mass_ratio)
# distribution_p.initialize(grid=grid_p, vt=vt_p, drift=0, perturbation=False)
# distribution_p.fourier_transform(), distribution_p.inverse_fourier_transform()
#
# # test elliptic solver
# elliptic = ell.Elliptic(resolution=elements[0])
# elliptic.poisson_solve(distribution_e=distribution_e, distribution_p=distribution_p, grids=grids)
#
# plotter_e = my_plt.Plotter(grid=grid_e)
# plotter_e.distribution_contourf(distribution=distribution_e, plot_spectrum=True)  # , remove_average=True)
# plotter_e.spatial_scalar_plot(scalar=distribution_e.zero_moment, y_axis='Zero moment electrons')
# # plotter_e.spatial_scalar_plot(scalar=elliptic.field, y_axis='Electric Field')
# plotter_e.show()
#
# plotter_p = my_plt.Plotter(grid=grid_p)
# plotter_p.distribution_contourf(distribution=distribution_p, plot_spectrum=False)
# # plotter_p.spatial_scalar_plot(scalar=distribution_p.zero_moment, y_axis='Zero moment protons')
# plotter_p.show()
