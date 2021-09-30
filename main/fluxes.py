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
        self.boundary_slices_pad = [(slice(self.x_res), slice(self.v_res + 2), 0),
                                    (slice(self.x_res), slice(self.v_res + 2), -1)]
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
        self.compute_flux(distribution=distribution, elliptic=elliptic, grid=grid)
        self.output.arr = (grid.v.J * self.v_flux_lgl(grid=grid) +
                           self.source_term_lgl(distribution=distribution, grid=grid))
        # return self.output.arr
        # if not gl:
        #     self.output.arr = (grid.v.J * self.v_flux_lgl(grid=grid))
        # self.output.arr = self.source_term(distribution=distribution, grid=grid)

    def compute_flux(self, distribution, elliptic, grid):
        """ Compute the flux convolution(field, distribution) using pseudospectral method """
        # Zero-pad spectrum
        padded_field_spectrum = cp.pad(elliptic.field.arr_spectral, grid.x.pad_width)
        padded_dist_spectrum = cp.pad(distribution.arr, grid.x.pad_width)[:, grid.x.pad_width:-grid.x.pad_width,
                                                                             grid.x.pad_width:-grid.x.pad_width]
        # Pseudospectral product
        field_nodal = cp.real(cp.fft.ifft(cp.fft.fftshift(padded_field_spectrum, axes=0), norm='forward', axis=0))
        distr_nodal = cp.real(cp.fft.ifft(cp.fft.fftshift(padded_dist_spectrum, axes=0), norm='forward', axis=0))
        nodal_flux = cp.multiply(field_nodal[:, None, None], distr_nodal)
        self.flux.arr = cp.fft.fftshift(cp.fft.fft(
            nodal_flux, axis=0, norm='forward'), axes=0
        )[grid.x.pad_width:-grid.x.pad_width, :, :]

    def v_flux_lgl(self, grid):
        return -1.0 * (basis_product(flux=self.flux.arr, basis_arr=grid.v.local_basis.internal, axis=2) -
                       self.numerical_flux_lgl(grid=grid))

    def numerical_flux_lgl(self, grid):
        # Allocate
        num_flux = cp.zeros(self.num_flux_size) + 0j

        # set padded flux
        padded_flux = cp.zeros((self.x_res, self.v_res + 2, self.order)) + 0j
        padded_flux[:, 1:-1, :] = self.flux.arr
        padded_flux[:, 0, -1] = 0.0  # -self.flux.arr[:, 0, 0]
        padded_flux[:, -1, 0] = 0.0  # -self.flux.arr[:, -1, 0]

        # Compute a central flux
        num_flux[self.boundary_slices[0]] = -1.0 * (cp.roll(padded_flux[self.boundary_slices_pad[1]],
                                                            shift=+1, axis=1)[:, 1:-1] +
                                                    self.flux.arr[self.boundary_slices[0]]) / 2.0
        num_flux[self.boundary_slices[1]] = (cp.roll(padded_flux[self.boundary_slices_pad[0]],
                                                     shift=-1, axis=1)[:, 1:-1] +
                                             self.flux.arr[self.boundary_slices[1]]) / 2.0

        return basis_product(flux=num_flux, basis_arr=grid.v.local_basis.numerical, axis=2)

    def source_term_lgl(self, distribution, grid):
        return -1.0j * cp.multiply(grid.x.device_wavenumbers[:, None, None],
                                   cp.einsum('ijk,mik->mij', grid.v.translation_matrix, distribution.arr))

    # def source_term_gl(self, distribution, grid):
    #     return -1.0j * cp.multiply(grid.x.device_wavenumbers[:, None, None],
    #                                cp.multiply(grid.v.device_arr[None, :, :], distribution.arr))
    #
    # def numerical_flux_gl(self, grid):
    #     # Allocate
    #     num_flux = cp.zeros(self.num_flux_size) + 0j
    #     boundary_flux = cp.zeros((self.x_res, self.v_res + 2, 2)) + 0j
    #
    #     # Accumulate on boundaries
    #     print(self.flux.arr.shape)
    #     print(grid.v.local_basis.boundary_accumulator.shape)
    #     left_nodes = cp.tensordot(self.flux.arr,
    #                                              grid.v.local_basis.boundary_accumulator[0, :],
    #                                              axes=([2], [0]))
    #     print(left_nodes.shape)
    #     quit()
    #     boundary_flux[:, 1:-1, 0] = cp.tensordot(self.flux.arr,
    #                                              grid.v.local_basis.boundary_accumulator[0, :],
    #                                              axes=([2], [0]))
    #     boundary_flux[:, 1:-1, 1] = cp.tensordot(self.flux.arr,
    #                                              grid.v.local_basis.boundary_accumulator[1, :],
    #                                              axes=([2], [0]))
    #     print(self.flux.arr[2, 9, :])
    #     print(boundary_flux[2, 10, :])
    #
    #     # compute a central flux
    #     num_flux[:, :, 0] = -1.0 * (cp.roll(boundary_flux[:, :, 1], shift=+1, axis=1)[:, 1:-1] +
    #                                 boundary_flux[:, 1:-1, 0]) / 2.0
    #     num_flux[:, :, 1] = (cp.roll(boundary_flux[:, :, 0], shift=-1, axis=1)[:, 1:-1] +
    #                          boundary_flux[:, 1:-1, 1]) / 2.0
    #     print(num_flux[2, 9, :])
    #
    #     output = basis_product(flux=num_flux, basis_arr=grid.v.local_basis.numerical, axis=2)
    #     # print(output[2, 9, :])
    #     return output
    #
    #
    # def v_flux_gl(self, grid):
    #     internal = basis_product(flux=self.flux.arr, basis_arr=grid.v.local_basis.internal, axis=2) * grid.v.J
    #     numerical = self.numerical_flux_gl(grid=grid)
    #     print(internal[2, 9, :])
    #     print(numerical[2, 9, :])
    #     print(grid.v.J)
    #     quit()
    #     return -1.0 * (basis_product(flux=self.flux.arr, basis_arr=grid.v.local_basis.internal, axis=2) -
    #                    self.numerical_flux_gl(grid=grid))
