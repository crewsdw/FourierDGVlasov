import numpy as np
import grid as g
import variables as var
import elliptic as ell
import plotter as my_plt
import data
import cupy as cp

# elements and order
elements, order = [4000, 100], 8
vt = 1
chi = 0.05
vb = 5
vtb = chi ** (1 / 3) * vb

# set up grids
length = 2000
lows = np.array([-length / 2, -5 * vt])
highs = np.array([length / 2, 15 * vt])
Grid = g.PhaseSpace(lows=lows, highs=highs, elements=elements, order=order)

# Read data
DataFile = data.Data(folder='..\\bot\\', filename='bot_file_test_4')
time_data, distribution_data, density_data, field_data = DataFile.read_file()

Plotter = my_plt.Plotter(grid=Grid)

# Unpack data, distribution
for idx, time in enumerate(time_data):
    print('Data at time {:0.3e}'.format(time))
    Distribution = var.Distribution(resolutions=elements, order=order, charge_mass=-1.0)
    Distribution.arr_nodal = cp.asarray(distribution_data[idx])
    Distribution.zero_moment.arr_nodal = cp.asarray(density_data[idx])
    Distribution.fourier_transform(), Distribution.zero_moment.fourier_transform()
    # Field
    Elliptic = ell.Elliptic(resolution=elements[0])
    Elliptic.field.arr_nodal = cp.asarray(field_data[idx])
    Elliptic.field.fourier_transform()

    # Analyze data
    Plotter.distribution_contourf(distribution=Distribution, remove_average=True)
    Plotter.show()

print('Done')
