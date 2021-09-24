# import cupy as np
import numpy as np
import variables as var


class Elliptic:
    def __init__(self, resolution):
        self.potential = var.SpaceScalar(resolution=resolution)
        self.field = var.SpaceScalar(resolution=resolution)

    def poisson_solve(self, distribution, grid, invert=True):
        # Compute zeroth moment, integrate(c_n(v)dv)
        distribution.compute_zero_moment(grid=grid)

        # Adjust for charge neutrality
        distribution.zero_moment.arr_spectral[grid.x.zero_idx] -= 1.0

        # Compute field spectrum
        self.field.arr_spectral = 1j * np.nan_to_num(np.divide(distribution.zero_moment.arr_spectral,
                                                               grid.x.device_wavenumbers,
                                                               where=grid.x.device_wavenumbers != 0))

        if invert:
            self.field.inverse_fourier_transform()
