import numpy as np
import cupy as cp


class SpaceScalar:
    def __init__(self, resolution):
        self.res = resolution
        self.arr_nodal, self.arr_spectral = None, None

    def fourier_transform(self):
        self.arr_spectral = cp.fft.fftshift(cp.fft.fft(self.arr_nodal, norm='forward'))

    def inverse_fourier_transform(self):
        self.arr_nodal = cp.real(cp.fft.ifft(cp.fft.fftshift(self.arr_spectral), norm='forward'))


class Distribution:
    def __init__(self, resolutions, order):
        self.x_res, self.v_res = resolutions
        self.order = order

        # arrays
        self.arr, self.arr_nodal = None, None
        self.zero_moment = SpaceScalar(resolution=resolutions[0])

    def compute_zero_moment(self, grid):
        self.zero_moment.arr_spectral = grid.v.zero_moment(function=self.arr, idx=[1, 2])

    def grid_flatten(self):
        return self.arr_nodal.reshape(self.x_res, self.v_res * self.order)

    def spectral_flatten(self):
        return self.arr.reshape(self.arr.shape[0], self.v_res * self.order)

    def initialize(self, grid):
        ix, iv = cp.ones_like(grid.x.device_arr), cp.ones_like(grid.v.device_arr)
        maxwellian = cp.tensordot(ix, grid.v.compute_maxwellian(thermal_velocity=1.0,
                                                                drift_velocity=0.0), axes=0)

        # compute perturbation
        perturbation = (grid.eigenfunction(thermal_velocity=1,
                                          drift_velocity=0.0,
                                          eigenvalue=1.68 - 0.4j) +
                        grid.eigenfunction(thermal_velocity=1,
                                           drift_velocity=0.0,
                                           eigenvalue=-1.68 - 0.4j))
        self.arr_nodal = maxwellian + 1.0e-2 * perturbation

    def fourier_transform(self):
        self.arr = cp.fft.fftshift(cp.fft.fft(self.arr_nodal, axis=0, norm='forward'), axes=0)

    def inverse_fourier_transform(self):
        self.arr_nodal = cp.real(cp.fft.ifft(cp.fftshift(self.arr), norm='forward', axis=0))
