import numpy as np
import grid as g
import variables as var
import elliptic as ell
import plotter as my_plt
import data
import cupy as cp

# elements and order
elements, order = [5000, 20], 25  # [5000, 50], 8
vt = 1
chi = 0.05
vb = 5
vtb = chi ** (1 / 3) * vb

# set up grids
length = 1000
lows = np.array([-length / 2, -15 * vt])
highs = np.array([length / 2, 15 * vt])
Grid = g.PhaseSpace(lows=lows, highs=highs, elements=elements, order=order)

# Read data
DataFile = data.Data(folder='..\\bot\\', filename='bot_file_test_150')
time_data, distribution_data, density_data, field_data, total_eng, total_den = DataFile.read_file()

# Set up plotter
Plotter = my_plt.Plotter(grid=Grid)

# Loop through time data
avg_dists = np.zeros((time_data.shape[0], elements[1], order))
covariance = np.zeros_like(avg_dists)
field_psd = np.zeros((time_data.shape[0], elements[0]//2 + 1))
avg_grads = np.zeros_like(avg_dists)
diff_estimate = np.zeros_like(avg_dists)
variance_of_correlation = np.zeros_like(avg_dists)
delta_of_correlation = np.zeros((time_data.shape[0], elements[0], elements[1], order))

# just_some_idx, just_some_time = np.array([1, 5, 10, 15]), np.array([1, 5, 10, 15])
jump = 0
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
    avg_grads[idx, :, :] = Distribution.compute_average_gradient(grid=Grid)
    field_psd[idx, :] = np.absolute(Elliptic.field.arr_spectral.flatten().get())**2.0
    diff_estimate[idx, :, :] = (covariance[idx, :, :]) / (avg_grads[idx, :, :])
    delta_of_correlation[idx, :, :, :] = (Distribution.delta_f * Elliptic.field.arr_nodal[:, None, None].get() -
                                          covariance[idx, :, :])
    DeltaOfCorrelation = var.Distribution(resolutions=elements, order=order, charge_mass=-1.0)
    DeltaOfCorrelation.arr_nodal = cp.asarray(delta_of_correlation[idx, :, :, :])
    variance_of_correlation[idx, :, :] = Distribution.variance_of_field_particle_covariance(Elliptic=Elliptic,
                                                                                            Grid=Grid,
                                                                                            covariance=covariance[idx, :, :])
    # diff_estimate[idx, :, :] = avg_grads[idx, :, :] / (covariance[idx, :, :]+1.0e-6)  # ) / (avg_grads[idx, :, :])
    # diff_estimate[diff_estimate < 0] = 0
    # diff_estimate[diff_estimate > 3] = 0
    # Analyze data
    # Plotter.distribution_contourf(distribution=Distribution, plot_spectrum=False, remove_average=True,
    #                               max_cb=0.005, save='..\\bot_figs\\pdf{:d}'.format(int(time)))
    # Plotter.distribution_contourf(distribution=DeltaOfCorrelation, plot_spectrum=False, remove_average=False,
    #                               max_cb=None, save='..\\bot_figs\\dropped_term_{:d}'.format(int(time)))
    # Plotter.plot_average_distribution(distribution=Distribution)
    # Plotter.show()

Plotter.plot_many_velocity_averages(time_data, avg_dists, y_label='Average distribution')
Plotter.plot_many_velocity_averages(time_data, covariance,
                                    y_label=r'Field-particle covariance, $\langle \Delta(f)E\rangle_L$')
Plotter.plot_many_velocity_averages(time_data, avg_grads, y_label='Gradient of average distribution')
Plotter.plot_many_velocity_averages(time_data, diff_estimate, y_label='Estimate of diffusion coefficient')
Plotter.plot_many_velocity_averages(time_data, variance_of_correlation,
                                    y_label=r'Variance of correlation, $\langle\langle \Delta(f)E\rangle\rangle_L$')
Plotter.plot_many_field_power_spectra(time_data, field_psd)
print(total_eng.shape)
# Plotter.time_series_plot(time_in=time_data, series_in=total_eng,
#                          y_axis='Total energy', log=False, numpy=True)
Plotter.show()

print('Done')
