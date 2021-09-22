import numpy as np
import cupy as cp


def basis_product(flux, basis_arr, axis, permutation):
    return cp.tensordot(flux, basis_arr,
                        axes=([axis], [1]))


class DGFlux:
    def __init__(self, resolution, order):
        self.resolution = resolution
        self.order = order

        # slices into the DG boundaries (list of tuples)
        self.boundary_slices = [(slice(resolution), 0), (slice(resolution), -1)]
        # self.flux_slice = [(slice(resolution), slice(order))]  # not necessary
        self.num_flux_size = [(resolution, 2)]

        # arrays
        self.flux = None

    def semi_discrete_rhs(self, distribution, elliptic, grid):
        self.compute_flux()
        return grid.v.J * self.v_flux(function=function, )


    def compute_flux(self, distribution, elliptic):
        """ Compute the flux convolution(field, distribution) using pseudospectral method """
        self.flux = cp.fft.fftshift(
            cp.fft.fft(cp.multiply(elliptic.field.inverse_fourier_transform()[:, None, None],
                                   distribution.inverse_fourier_transform()),
                       norm='forward')
        )


