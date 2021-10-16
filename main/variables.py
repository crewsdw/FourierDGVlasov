import numpy as np
import cupy as cp


class SpaceScalar:
    def __init__(self, resolution):
        self.res = resolution
        self.arr_nodal, self.arr_spectral = None, None

    def fourier_transform(self):
        # self.arr_spectral = cp.fft.fftshift(cp.fft.fft(self.arr_nodal, norm='forward'))
        self.arr_spectral = cp.fft.rfft(self.arr_nodal, norm='forward')

    def inverse_fourier_transform(self):
        # self.arr_nodal = cp.real(cp.fft.ifft(cp.fft.fftshift(self.arr_spectral), norm='forward'))
        self.arr_nodal = cp.fft.irfft(self.arr_spectral, norm='forward')

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
    def __init__(self, resolutions, order, charge_mass):
        self.x_res, self.v_res = resolutions
        self.order = order
        self.charge_mass = charge_mass

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

    def initialize(self, grid, vt, drift, perturbation=True):
        ix, iv = cp.ones_like(grid.x.device_arr), cp.ones_like(grid.v.device_arr)
        maxwellian = cp.tensordot(ix, grid.v.compute_maxwellian(thermal_velocity=vt,
                                                                drift_velocity=drift), axes=0)

        # compute perturbation
        # perturbation = cp.imag(grid.eigenfunction(thermal_velocity=1,
        #                                           drift_velocity=[2, -2],
        #                                           eigenvalue=1.20474886j))
        # print('The expected growth rate is {:0.3e}'.format(grid.x.fundamental * 1.2))
        # perturbation = np.real(grid.eigenfunction(thermal_velocity=1,
        #                                           drift_velocity=2.0,
        #                                           eigenvalue=-3.0j) +
        #                        grid.eigenfunction(thermal_velocity=1,
        #                                           drift_velocity=-2.0,
        #                                           eigenvalue=-3.0j)) / 2.0  # -1.68 - 0.4j
        if perturbation:
            # perturbation = np.multiply(np.sin(grid.x.fundamental * grid.x.device_arr)[:, None, None], maxwellian)
            perturbation = self.charge_mass * cp.real(grid.eigenfunction(thermal_velocity=vt,
                                                                         drift_velocity=drift,
                                                                         beams='one',
                                                                         # eigenvalue=+1.41575189 - 0.15329189j))
                                                                         eigenvalue=0.3403289 + 0.12783108j))
                                                                         # 1.43268952 - 0.1410875j))
                                                                         #  eigenvalue=0.17483063 - 0.0238957j))
            # perturbation += self.charge_mass * cp.real(grid.eigenfunction(thermal_velocity=vt,
            #                                                              drift_velocity=drift,
            #                                                              beams='one',
            #                                                              eigenvalue=-1.41575189 - 0.15329189j))
        else:
            perturbation = 0
        # grid.v.compute_maxwellian(thermal_velocity=1.0,
        #                           drift_velocity=0.0),
        # axes=0)
        # self.arr_nodal = maxwellian + 1.0e-7 * perturbation
        self.arr_nodal = maxwellian + 1.0e-1 * perturbation

    def fourier_transform(self):
        # self.arr = cp.fft.fftshift(cp.fft.fft(self.arr_nodal, axis=0, norm='forward'), axes=0)
        self.arr = cp.fft.rfft(self.arr_nodal, axis=0, norm='forward')

    def inverse_fourier_transform(self):
        # self.arr_nodal = cp.real(cp.fft.ifft(cp.fft.fftshift(self.arr, axes=0), norm='forward', axis=0))
        self.arr_nodal = cp.fft.irfft(self.arr, axis=0, norm='forward')


def trapz(y, dx):
    """ Custom trapz routine using cupy """
    return cp.sum(y[:-1] + y[1:]) * dx / 2.0
