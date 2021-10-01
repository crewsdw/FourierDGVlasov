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

    def integrate(self, grid):
        arr_add = cp.append(self.arr_nodal, self.arr_nodal[0])
        # x_add = cp.append(grid.x.device_arr, grid.x.device_arr[-1] + grid.x.dx)
        return trapz(arr_add, grid.x.dx)

    def integrate_energy(self, grid):
        arr = 0.5 * self.arr_nodal ** 2.0
        arr_add = cp.append(arr, arr[0])
        # x_add = cp.append(grid.x.device_arr, grid.x.device_arr[-1] + grid.x.dx)
        return trapz(arr_add, grid.x.dx)


class Distribution:
    def __init__(self, resolutions, order):
        self.x_res, self.v_res = resolutions
        self.order = order

        # arrays
        self.arr, self.arr_nodal = None, None
        self.zero_moment = SpaceScalar(resolution=resolutions[0])
        self.second_moment = SpaceScalar(resolution=resolutions[0])

    def compute_zero_moment(self, grid):
        self.inverse_fourier_transform()
        self.zero_moment.arr_nodal = grid.v.zero_moment(function=self.arr_nodal, idx=[1, 2])
        self.zero_moment.fourier_transform()
        # self.zero_moment.arr_spectral = grid.v.zero_moment(function=self.arr, idx=[1, 2])

    def total_thermal_energy(self, grid):
        self.inverse_fourier_transform()
        self.second_moment.arr_nodal = grid.v.second_moment(function=self.arr_nodal, idx=[1, 2])
        return 0.5 * self.second_moment.integrate(grid=grid)

    def total_density(self, grid):
        self.inverse_fourier_transform()
        self.compute_zero_moment(grid=grid)
        return self.zero_moment.integrate(grid=grid)

    def grid_flatten(self):
        return self.arr_nodal.reshape(self.x_res, self.v_res * self.order)

    def spectral_flatten(self):
        return self.arr.reshape(self.arr.shape[0], self.v_res * self.order)

    def initialize(self, grid):
        ix, iv = cp.ones_like(grid.x.device_arr), cp.ones_like(grid.v.device_arr)
        maxwellian = 0.5 * (cp.tensordot(ix, grid.v.compute_maxwellian(thermal_velocity=1.0,
                                                                       drift_velocity=0.0), axes=0) +
                            cp.tensordot(ix, grid.v.compute_maxwellian(thermal_velocity=1.0,
                                                                       drift_velocity=0.0), axes=0))

        # compute perturbation
        perturbation = cp.imag(grid.eigenfunction(thermal_velocity=1,
                                                  drift_velocity=[2, -2],
                                                  eigenvalue=1.20474886j))
        # print('The expected growth rate is {:0.3e}'.format(grid.x.fundamental * 1.2))
        # perturbation = np.real(grid.eigenfunction(thermal_velocity=1,
        #                                           drift_velocity=2.0,
        #                                           eigenvalue=-3.0j) +
        #                        grid.eigenfunction(thermal_velocity=1,
        #                                           drift_velocity=-2.0,
        #                                           eigenvalue=-3.0j)) / 2.0  # -1.68 - 0.4j
        perturbation = np.multiply(np.sin(grid.x.fundamental * grid.x.device_arr)[:, None, None], maxwellian)
        # grid.v.compute_maxwellian(thermal_velocity=1.0,
        #                           drift_velocity=0.0),
        # axes=0)
        # self.arr_nodal = maxwellian + 1.0e-7 * perturbation
        self.arr_nodal = maxwellian + 5.0e-1 * perturbation

    def fourier_transform(self):
        self.arr = cp.fft.fftshift(cp.fft.fft(self.arr_nodal, axis=0, norm='forward'), axes=0)

    def inverse_fourier_transform(self):
        self.arr_nodal = cp.real(cp.fft.ifft(cp.fft.fftshift(self.arr, axes=0), norm='forward', axis=0))


def trapz(y, dx):
    """ Custom trapz routine using cupy """
    return cp.sum(y[:-1] + y[1:]) * dx / 2.0
