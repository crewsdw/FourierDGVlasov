import numpy as np
import grid as g
import variables as var
import elliptic as ell
import plotter as my_plt
import data
import cupy as cp
import matplotlib.pyplot as plt
import dielectric

# elements and order
elements, order = [5000, 50], 10  # [20000, 25], 20  # [5000, 50], 8
vt = 1
chi = 0.05
vb = 5
vtb = chi ** (1 / 3) * vb

# set up grids
length = 5000
lows = np.array([-length / 2, -25 * vt])
highs = np.array([length / 2, 25 * vt])
Grid = g.PhaseSpace(lows=lows, highs=highs, elements=elements, order=order)

# Read data
# DataFile = data.Data(folder='..\\ts\\', filename='two_stream_test30')
# DataFile = data.Data(folder='..\\bot\\', filename='bot_1000LD_t131')
DataFile = data.Data(folder='..\\bot\\', filename='bot_5000LD_t101')
time_data, distribution_data, density_data, field_data, total_eng, total_den = DataFile.read_file()

# Set up plotter
Plotter = my_plt.Plotter(grid=Grid)

# Loop through time data
avg_dists = np.zeros((time_data.shape[0], elements[1], order))
avg_avg_dists = np.zeros((elements[1], order))
covariance = np.zeros_like(avg_dists)
field_psd = np.zeros((time_data.shape[0], elements[0] // 2 + 1))
avg_grads = np.zeros_like(avg_dists)
diff_estimate = np.zeros_like(avg_dists)
variance_of_correlation = np.zeros_like(avg_dists)
delta_of_correlation = np.zeros((time_data.shape[0], elements[0], elements[1], order))

# just_some_idx, just_some_time = np.array([1, 5, 10, 15]), np.array([1, 5, 10, 15])
# Plotter.plot_two_time_correlation_function(distribution_data=distribution_data[1:], time_data=time_data[1:],
#                                            elements=elements, order=order)
# Compute mean-square bounce frequency
# rms_bounce_freq = np.zeros(time_data.shape[0])
# for idx, time in enumerate(time_data):
#     delta_n = np.abs(density_data[idx, :] - 1)
#     rms_bounce_freq[idx] = np.sum(delta_n[:-1] + delta_n[1:], axis=0) * (Grid.x.arr[1] - Grid.x.arr[0]) / 2.0 / length
# # print(rms_bounce_freq)
# print('The bounce frequencies are')
# print(np.sqrt(rms_bounce_freq))
# quit()

# Plot autocorrelation function
# Plotter.plot_autocorrelation_function(distribution_data=distribution_data[1:], time_data=time_data[1:],
#                                       elements=elements, order=order)
# Plotter.wavepacket_autocorrelation(field_data=field_data[1:], time_data=time_data[1:], elements=elements)

jump = 2
for idx, time in enumerate(time_data[jump:]):
    # for iidx, time in enumerate(just_some_time):
    idx += jump
    # idx = just_some_idx[iidx]
    print('Data at time {:0.3e}'.format(time))
    # Unpack data, distribution, density
    Distribution = var.Distribution(resolutions=elements, order=order, charge_mass=-1.0)
    Distribution.arr_nodal = cp.asarray(distribution_data[idx])
    Distribution.zero_moment.arr_nodal = cp.asarray(density_data[idx])
    Distribution.fourier_transform(), Distribution.zero_moment.fourier_transform()
    # Field
    Elliptic = ell.Elliptic(resolution=elements[0])
    Elliptic.field.arr_nodal = cp.asarray(field_data[idx])
    Elliptic.field.fourier_transform()

    # Compute spatially-averaged distribution
    Distribution.average_distribution(grid=Grid)
    Distribution.compute_delta_f()
    covariance[idx, :, :] = Distribution.field_particle_covariance(Elliptic=Elliptic, Grid=Grid)

    avg_dists[idx, :, :] = Distribution.avg_dist
    if idx > 0:
        avg_avg_dists += Distribution.avg_dist
    avg_grads[idx, :, :] = Distribution.compute_average_gradient(grid=Grid)
    field_psd[idx, :] = np.absolute(Elliptic.field.arr_spectral.flatten().get()) ** 2.0
    diff_estimate[idx, :, :] = (covariance[idx, :, :]) / (avg_grads[idx, :, :])
    delta_of_correlation[idx, :, :, :] = (Distribution.delta_f * Elliptic.field.arr_nodal[:, None, None].get() -
                                          covariance[idx, :, :])
    DeltaOfCorrelation = var.Distribution(resolutions=elements, order=order, charge_mass=-1.0)
    DeltaOfCorrelation.arr_nodal = cp.asarray(delta_of_correlation[idx, :, :, :])
    variance_of_correlation[idx, :, :] = Distribution.variance_of_field_particle_covariance(Elliptic=Elliptic,
                                                                                            Grid=Grid,
                                                                                            covariance=covariance[idx,
                                                                                                       :, :])
    # diff_estimate[idx, :, :] = avg_grads[idx, :, :] / (covariance[idx, :, :]+1.0e-6)  # ) / (avg_grads[idx, :, :])
    # diff_estimate[diff_estimate < 0] = 0
    # diff_estimate[diff_estimate > 3] = 0
    # Analyze data
    # Plotter.spatial_scalar_plot(scalar=Elliptic.field, y_axis=r'Electric field $E(x)$', spectrum=False,
    #                             title='{:d}'.format(int(10*time)),
    #                             save='..\\bot_figs\\field{:d}'.format(int(10 * time)))
    #
    # plt.close()
    # Plotter.show()
    # Plotter.distribution_contourf(distribution=Distribution, plot_spectrum=False, remove_average=True,
    #                               max_cb=None, save='..\\bot_figs\\1000L\\pdf{:d}'.format(int(10 * time)))
    # Plotter.show()
    # plt.close()
    # Plotter.distribution_contourf(distribution=DeltaOfCorrelation, plot_spectrum=False, remove_average=False,
    #                               max_cb=None, save='..\\bot_figs\\dropped_term_{:d}'.format(int(time)))
    # Plotter.distribution_contourf(distribution=Distribution, plot_spectrum=True, remove_average=True,
    #                               max_cb=None)
    # Plotter.plot_average_distribution(distribution=Distribution)
    # Plotter.show()

Plotter.distribution_contourf(distribution=Distribution, plot_spectrum=True, remove_average=True,
                              max_cb=None)
Plotter.show()

# Average of average distributions
avg_avg_dists = avg_avg_dists / idx
ensemble_average = var.Scalar(resolution=elements[1], order=order)
ensemble_average.arr = avg_avg_dists
# Compute dielectric function solution
grid_k = Grid.x.wavenumbers[(0.2 <= Grid.x.wavenumbers) & (Grid.x.wavenumbers <= 0.35)]
dielectric.solve_approximate_dielectric_function(distribution=ensemble_average, grid_v=Grid.v, grid_k=grid_k)

# Estimated relaxation time
relax_time = np.zeros_like(avg_dists)
dt = 5  # time_data[2] - time_data[0]
# print(dt)
relax_time[1:-1, :, :] = np.abs(avg_dists[2:, :, :] - avg_dists[:-2, :, :]) / (2 * dt) / avg_dists[1:-1, :, :]
relax_time[0, :, :] = np.abs(avg_dists[1, :, :] - avg_dists[0, :, :]) / dt / avg_dists[0, :, :]
relax_time[-1, :, :] = np.abs(avg_dists[-1, :, :] - avg_dists[-2, :, :]) / dt / avg_dists[-1, :, :]
relax_time[:, :5, :] = 0
relax_time[:, -5:, :] = 0

# quit()
# plt.figure()
# plt.plot(Grid.v.arr.flatten(), avg_avg_dists.flatten(), 'o--',
#          label=r'Time-averaged $\langle f\rangle_L$ at saturation')
# plt.plot(Grid.v.arr.flatten(), avg_dists[0, :, :].flatten(), 'o--', label=r'Initial $\langle f\rangle_L$')
# plt.grid(True), plt.xlabel('Velocity'), plt.ylabel('Average distribution function')
# plt.legend(loc='best'), plt.tight_layout()
#
# plt.figure()
# delta_f = avg_avg_dists - avg_dists[0, :, :]
# plt.plot(Grid.v.arr.flatten(), delta_f.flatten(), 'o--')
# plt.grid(True), plt.xlabel('Velocity'), plt.ylabel(r'Change in $\langle f\rangle_L$')
# plt.legend(loc='best'), plt.tight_layout()

# Integrate and predict saturated state
# approx_field_distribution = var.Distribution(resolutions=elements, order=order, charge_mass=-1.0)
# approx_field_distribution.arr_nodal = np.zeros_like(delta_f)
# for idx1 in range(10, 35):
#     approx_field_distribution.arr_nodal[idx1, :] = ((Grid.v.arr[idx1, :] ** 3.0) *
#                                                     np.tensordot(Grid.v.global_quads[9:idx1, :].get() /
#                                                                  Grid.v.J_host[9:idx1, None],
#                                                                  delta_f[9:idx1, :],
#                                                                  axes=([0, 1], [0, 1])))
#
# plt.figure()
# plt.plot(Grid.v.arr.flatten(), approx_field_distribution.arr_nodal.flatten(), 'o--')
# plt.grid(True), plt.xlabel('Velocity'), plt.ylabel(r'Saturated spectrum $E_v^2$')
# plt.legend(loc='best'), plt.tight_layout()
#
# plt.show()

Plotter.plot_many_velocity_averages(time_data, relax_time, y_label=r'Relaxation frequency $\tau_r^{-1}$')
Plotter.plot_many_velocity_averages(time_data, np.log(avg_dists), y_label=r'Average distribution $\langle f\rangle_L$')
Plotter.plot_many_velocity_averages(time_data, covariance,
                                    y_label=r'Field-particle covariance, $\langle\delta f E\rangle_L$')
Plotter.plot_many_velocity_averages(time_data, avg_grads, y_label='Gradient of average distribution')
Plotter.plot_many_velocity_averages(time_data, diff_estimate, y_label=r'DNS diffusivity $D(v)$')
Plotter.plot_many_velocity_averages(time_data, variance_of_correlation,
                                    y_label=r'Variance of field-particle correlation, '
                                            r'$\langle\langle f_1 E\rangle\rangle_L$')
# Plotter.plot_average_field_power_spectrum(time_data, field_psd, start_idx=2)
# Plotter.plot_many_field_power_spectra(time_data, field_psd)
print(total_eng.shape)
# Plotter.time_series_plot(time_in=time_data, series_in=total_eng,
#                          y_axis='Total energy', log=False, numpy=True)
Plotter.show()

print('Done')
