import numpy as np
# import cupy as np
import variables as var


def basis_product(flux, basis_arr, axis):
    return np.tensordot(flux, basis_arr,
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
        # two-thirds rule
        # distribution.arr[:grid.x.two_thirds_low, :, :] = 0.0 + 0j
        # distribution.arr[grid.x.two_thirds_high:, :, :] = 0.0 + 0j
        """ Computes the semi-discrete equation """
        # Do elliptic problem
        elliptic.poisson_solve(distribution=distribution, grid=grid, invert=False)
        # elliptic.field.arr_spectral[:grid.x.two_thirds_low] = 0.0 + 0j
        # elliptic.field.arr_spectral[grid.x.two_thirds_high:] = 0.0 + 0j
        # Compute the flux
        self.compute_flux(distribution=distribution, elliptic=elliptic, grid=grid)
        self.output.arr = (grid.v.J * self.v_flux(grid=grid)
                           + self.source_term(distribution=distribution, grid=grid))
        return self.output.arr
        # self.output.arr = self.source_term(distribution=distribution, grid=grid)

    def compute_flux(self, distribution, elliptic, grid):
        """ Compute the flux convolution(field, distribution) using pseudospectral method """
        # elliptic.poisson_solve(distr)
        # elliptic.field.inverse_fourier_transform()
        # distribution.inverse_fourier_transform()
        # print(elliptic.field.arr_spectral)
        padded_field_spectrum = np.pad(elliptic.field.arr_spectral, grid.x.pad_width)
        padded_dist_spectrum = np.pad(distribution.arr, grid.x.pad_width)[:,
                               grid.x.pad_width:-grid.x.pad_width,
                               grid.x.pad_width:-grid.x.pad_width]
        # print(distribution.arr.shape)
        # print(padded_dist_spectrum.shape)
        # quit()
        # pseudospectral product
        field_nodal = np.real(np.fft.ifft(np.fft.fftshift(padded_field_spectrum, axes=0), norm='forward', axis=0))
        distr_nodal = np.real(np.fft.ifft(np.fft.fftshift(padded_dist_spectrum, axes=0), norm='forward', axis=0))
        nodal_flux = np.multiply(field_nodal[:, None, None], distr_nodal)
        self.flux.arr = np.fft.fftshift(np.fft.fft(
            nodal_flux, axis=0, norm='forward'), axes=0
        )[grid.x.pad_width:-grid.x.pad_width, :, :]
        # self.flux.fourier_transform()[grid.x.pad_width]

    def v_flux(self, grid):
        return -1.0 * (basis_product(flux=self.flux.arr, basis_arr=grid.v.local_basis.internal, axis=2) -
                       self.numerical_flux(grid=grid))

    def numerical_flux(self, grid):
        # Allocate
        num_flux = np.zeros(self.num_flux_size) + 0j

        # set padded flux
        padded_flux = np.zeros((self.x_res, self.v_res + 2, self.order)) + 0j
        padded_flux[:, 1:-1, :] = self.flux.arr
        padded_flux[:, 0, -1] = 0.0  # -self.flux.arr[:, 0, 0]
        padded_flux[:, -1, 0] = 0.0  # -self.flux.arr[:, -1, 0]

        # Compute a central flux
        num_flux[self.boundary_slices[0]] = -1.0 * (np.roll(padded_flux[self.boundary_slices_pad[1]],
                                                            shift=+1, axis=1)[:, 1:-1] +
                                                    self.flux.arr[self.boundary_slices[0]]) / 2.0
        num_flux[self.boundary_slices[1]] = (np.roll(padded_flux[self.boundary_slices_pad[0]],
                                                     shift=-1, axis=1)[:, 1:-1] +
                                             self.flux.arr[self.boundary_slices[1]]) / 2.0

        return basis_product(flux=num_flux, basis_arr=grid.v.local_basis.numerical, axis=2)

    def source_term(self, distribution, grid):
        # return -1.0j * np.multiply(grid.x.device_wavenumbers[:, None, None],
        #                            np.multiply(grid.v.device_arr[None, :, :], distribution.arr))
        return -1.0j * np.multiply(grid.x.device_wavenumbers[:, None, None],
                                   np.einsum('ijk,mik->mij', grid.v.translation_matrix, distribution.arr))
        # orig = np.multiply(grid.v.device_arr[None, :, :], distribution.arr)
        # # print(grid.v.translation_matrix.shape)
        # # print(distribution.arr.shape)
        # translation = np.einsum('ijk,mik->mij', grid.v.translation_matrix, distribution.arr)
        # print(orig[2, 10, :])
        # print(translation[2, 10, :])
        # print(np.matmul(grid.v.translation_matrix[10, :, :], distribution.arr[2, 10, :]))
        # # print(orig[16,10,:]-translation[16,10,:])
        # # quit()
