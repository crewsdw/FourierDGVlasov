import numpy as np


# import cupy as np


class SpaceScalar:
    def __init__(self, resolution):
        self.res = resolution
        self.arr_nodal, self.arr_spectral = None, None

    def fourier_transform(self):
        self.arr_spectral = np.fft.fftshift(np.fft.fft(self.arr_nodal, norm='forward'))

    def inverse_fourier_transform(self):
        self.arr_nodal = np.real(np.fft.ifft(np.fft.fftshift(self.arr_spectral), norm='forward'))


class Distribution:
    def __init__(self, resolutions, order):
        self.x_res, self.v_res = resolutions
        self.order = order

        # arrays
        self.arr, self.arr_nodal = None, None
        self.zero_moment = SpaceScalar(resolution=resolutions[0])

    def compute_zero_moment(self, grid):
        self.inverse_fourier_transform()
        self.zero_moment.arr_nodal = grid.v.zero_moment(function=self.arr_nodal, idx=[1, 2])
        self.zero_moment.fourier_transform()
        # self.zero_moment.arr_spectral = grid.v.zero_moment(function=self.arr, idx=[1, 2])

    def grid_flatten(self):
        return self.arr_nodal.reshape(self.x_res, self.v_res * self.order)

    def spectral_flatten(self):
        return self.arr.reshape(self.arr.shape[0], self.v_res * self.order)

    def initialize(self, grid):
        ix, iv = np.ones_like(grid.x.device_arr), np.ones_like(grid.v.device_arr)
        maxwellian = 0.5 * (np.tensordot(ix, grid.v.compute_maxwellian(thermal_velocity=1.0,
                                                                       drift_velocity=2.0), axes=0) +
                            np.tensordot(ix, grid.v.compute_maxwellian(thermal_velocity=1.0,
                                                                       drift_velocity=-2.0), axes=0))

        # compute perturbation
        perturbation = np.imag(grid.eigenfunction(thermal_velocity=1,
                                                  drift_velocity=[2, -2],
                                                  eigenvalue=1.20474886j))
        # print('The expected growth rate is {:0.3e}'.format(grid.x.fundamental * 1.2))
        # perturbation = np.real(grid.eigenfunction(thermal_velocity=1,
        #                                           drift_velocity=2.0,
        #                                           eigenvalue=-3.0j) +
        #                        grid.eigenfunction(thermal_velocity=1,
        #                                           drift_velocity=-2.0,
        #                                           eigenvalue=-3.0j)) / 2.0  # -1.68 - 0.4j
        # perturbation = np.multiply(np.sin(grid.x.fundamental * grid.x.device_arr)[:, None, None], maxwellian)
        # grid.v.compute_maxwellian(thermal_velocity=1.0,
        #                           drift_velocity=0.0),
        # axes=0)
        self.arr_nodal = maxwellian + 1.0e-7 * perturbation

    def fourier_transform(self):
        self.arr = np.fft.fftshift(np.fft.fft(self.arr_nodal, axis=0, norm='forward'), axes=0)

    def inverse_fourier_transform(self):
        self.arr_nodal = np.real(np.fft.ifft(np.fft.fftshift(self.arr, axes=0), norm='forward', axis=0))
