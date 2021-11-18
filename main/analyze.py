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
length = 2000
lows = np.array([-length / 2, -13 * vt])
highs = np.array([length / 2, 13 * vt])
Grid = g.PhaseSpace(lows=lows, highs=highs, elements=elements, order=order)

# Read data
DataFile = data.Data(folder='..\\bot\\', filename='bot_file_test_150')
time_data, distribution_data, density_data, field_data, total_eng, total_den = DataFile.read_file()

# Set up plotter
Plotter = my_plt.Plotter(grid=Grid)

# Loop through time data
avg_dists = np.zeros((time_data.shape[0], elements[1], order))
field_psd = np.zeros((time_data.shape[0], elements[0]//2 + 1))
for idx, time in enumerate(time_data):
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
    Distribution.average_distribution()
    Distribution.compute_delta_f()

    avg_dists[idx, :, :] = Distribution.avg_dist
    field_psd[idx, :] = np.absolute(Elliptic.field.arr_spectral.flatten().get())**2.0
    # Analyze data
    Plotter.distribution_contourf(distribution=Distribution, plot_spectrum=False, remove_average=True,
                                  max_cb=0.005, save='..\\bot_figs\\pdf{:d}'.format(int(time)))
    # Plotter.plot_average_distribution(distribution=Distribution)
    # Plotter.show()

Plotter.plot_many_velocity_averages(time_data, avg_dists)
Plotter.plot_many_field_power_spectra(time_data, field_psd)
print(total_eng.shape)
# Plotter.time_series_plot(time_in=time_data, series_in=total_eng,
#                          y_axis='Total energy', log=False, numpy=True)
Plotter.show()

print('Done')
