import numpy as np
import cupy as cp
import variables as var


def basis_product(flux, basis_arr, axis):
    return cp.tensordot(flux, basis_arr,
                        axes=([axis], [1]))


class DGFlux:
    def __init__(self, resolutions, order):
        self.x_res, self.v_res = resolutions
        self.order = order

        # slices into the DG boundaries (list of tuples)
        self.boundary_slices = [(slice(self.x_res), slice(self.v_res), 0),
                                (slice(self.x_res), slice(self.v_res), -1)]
        # self.flux_slice = [(slice(resolution), slice(order))]  # not necessary
        self.num_flux_size = (self.x_res, self.v_res, 2)

        # arrays
        self.flux = var.Distribution(resolutions=resolutions, order=order)
        self.output = var.Distribution(resolutions=resolutions, order=order)

    def semi_discrete_rhs(self, distribution, elliptic, grid):
        """ Computes the semi-discrete equation """
        # Do elliptic problem
        elliptic.poisson_solve(distribution=distribution, grid=grid, invert=False)
        # Compute the flux
        self.compute_flux(distribution=distribution, elliptic=elliptic)
        self.output.arr = (grid.v.J * self.v_flux(grid=grid)
                           + self.source_term(distribution=distribution, grid=grid))
        # self.output.arr = self.source_term(distribution=distribution, grid=grid)

    def compute_flux(self, distribution, elliptic):
        """ Compute the flux convolution(field, distribution) using pseudospectral method """
        # elliptic.field.inverse_fourier_transform()
        # elliptic.poisson_solve(distr)
        distribution.inverse_fourier_transform()
        self.flux.arr_nodal = cp.multiply(elliptic.field.arr_nodal[:, None, None],
                                          distribution.arr_nodal)
        self.flux.fourier_transform()

    def v_flux(self, grid):
        return -1.0 * (basis_product(flux=self.flux.arr, basis_arr=grid.v.local_basis.internal, axis=2) -
                self.numerical_flux(grid=grid))

    def numerical_flux(self, grid):
        # Allocate
        num_flux = cp.zeros(self.num_flux_size) + 0j

        # Compute a central flux
        num_flux[self.boundary_slices[0]] = -1.0 * (cp.roll(self.flux.arr[self.boundary_slices[1]],
                                                            shift=1, axis=1) +
                                                    self.flux.arr[self.boundary_slices[0]]) / 2.0
        num_flux[self.boundary_slices[1]] = (cp.roll(self.flux.arr[self.boundary_slices[0]],
                                                     shift=-1, axis=1) +
                                             self.flux.arr[self.boundary_slices[1]]) / 2.0

        return basis_product(flux=num_flux, basis_arr=grid.v.local_basis.numerical, axis=2)

    def source_term(self, distribution, grid):
        return -1.0j * cp.multiply(grid.x.device_wavenumbers[:, None, None],
                                   cp.multiply(grid.v.device_arr[None, :, :], distribution.arr))
