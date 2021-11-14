import numpy as np
import grid as g
import variables as var
import elliptic as ell
import plotter as my_plt
import data

# elements and order
elements, order = [4000, 600], 8
vt = 1
chi = 0.05
vb = 5
vtb = chi ** (1 / 3) * vb

# set up grids
length = 2000
lows = np.array([-length / 2, -8 * vt])
highs = np.array([length / 2, 17 * vt])
grid = g.PhaseSpace(lows=lows, highs=highs, elements=elements, order=order)

# Read data
DataFile = data.Data(folder='..\\bot\\', filename='bot_t3')
distribution_data, density_data, field_data = DataFile.read_file()

# Unpack data, distribution
distribution = var.Distribution(resolutions=elements, order=order, charge_mass=-1.0)
distribution.arr_nodal, distribution.zero_moment.arr_nodal = distribution_data, density_data
distribution.fourier_transform(), distribution.zero_moment.fourier_transform()
# Field
elliptic = ell.Elliptic(resolution=elements[0])
elliptic.field.arr_nodal = field_data
elliptic.field.fourier_transform()

# Analyze data

