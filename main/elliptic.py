import cupy as cp
import numpy as np
import variables as var


class Elliptic:
    def __init__(self, resolution):
        self.potential = var.SpaceScalar(resolution=resolution)
        self.field = var.SpaceScalar(resolution=resolution)

    def poisson_solve(self, distribution_e, distribution_p, grids, invert=True):
        # Compute zeroth moment, integrate(c_n(v)dv)
        distribution_e.compute_zero_moment(grid=grids[0])
        distribution_p.compute_zero_moment(grid=grids[1])

        # Adjust for charge neutrality
        # distribution_e.zero_moment.arr_spectral[grid.x.zero_idx] -= 1.0
        # distribution_p.zero_moment.arr_spectral[grid.x.zero_idx] -= 1.0

        # Compute field spectrum
        self.field.arr_spectral = -1j * cp.nan_to_num(cp.divide((distribution_p.zero_moment.arr_spectral -
                                                                 distribution_e.zero_moment.arr_spectral),
                                                                grids[0].x.device_wavenumbers))
        self.field.arr_spectral[grids[0].x.zero_idx] = 0 + 0j

        if invert:
            self.field.inverse_fourier_transform()

    def compute_field_energy(self, grid):
        self.field.inverse_fourier_transform()
        return self.field.integrate_energy(grid=grid)
        # return grid.x.compute_moment(function=0.5 * self.field.arr_nodal ** 2.0)
