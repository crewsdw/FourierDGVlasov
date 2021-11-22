import numpy as np
import cupy as cp
import tools.dispersion as dispersion
import scipy.optimize as opt
import cupyx.scipy.signal as sig
import matplotlib.pyplot as plt
from copy import deepcopy

cp.random.seed(1111)


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

    def compute_wigner_distribution(self, grid):
        # Zero-pad
        spectrum = cp.fft.fftshift(cp.fft.fft(self.arr_nodal))
        fourier_functions = (spectrum[None, :] *
                             cp.exp(1j * grid.x.device_wavenumbers[None, :] * grid.x.device_arr[:, None]))
        return sig.fftconvolve(cp.conj(fourier_functions), fourier_functions, mode='same', axes=1)


class Distribution:
    def __init__(self, resolutions, order, charge_mass):
        self.x_res, self.v_res = resolutions
        self.order = order
        self.charge_mass = charge_mass

        # arrays
        self.arr, self.arr_nodal = None, None
        self.zero_moment = SpaceScalar(resolution=resolutions[0])
        self.second_moment = SpaceScalar(resolution=resolutions[0])

        # post-processing attributes
        self.avg_dist, self.delta_f = None, None

    def compute_zero_moment(self, grid):
        self.inverse_fourier_transform()
        self.zero_moment.arr_nodal = grid.v.zero_moment(function=self.arr_nodal, idx=[1, 2])
        self.zero_moment.fourier_transform()
        # self.zero_moment.arr_spectral = grid.v.zero_moment(function=self.arr, idx=[1, 2])

    def total_thermal_energy(self, grid):
        self.inverse_fourier_transform()
        self.second_moment.arr_nodal = grid.v.second_moment(function=self.arr_nodal, idx=[1, 2])
        return 0.5 * self.second_moment.integrate(grid=grid)

    def average_distribution(self, grid):
        self.avg_dist = np.real(self.arr[0, :].get())
        # spectrum = np.tensordot(grid.v.fourier_quads, self.avg_dist, axes=([1, 2], [0, 1]))
        # deriv = 1j * grid.v.modes * spectrum
        # self.avg_dist = np.sum(spectrum[:, None, None] *
        #               np.exp(1j * grid.v.modes[:, None, None] * grid.v.arr[None, :, :]), axis=0)
        # self.inverse_fourier_transform()
        # self.avg_dist = trapz2(self.arr_nodal, grid.x.dx).get()

    def average_on_boundaries(self):
        self.arr_nodal[:, :, 0] = (self.arr_nodal[:, :, 0] + cp.roll(self.arr_nodal, shift=+1, axis=1)[:, :, -1]) / 2
        self.arr_nodal[:, :, -1] = (cp.roll(self.arr_nodal, shift=-1, axis=1)[:, :, 0] + self.arr_nodal[:, :, -1]) / 2

    def compute_delta_f(self):
        self.delta_f = self.arr_nodal.get() - self.avg_dist[None, :, :]

    def compute_average_gradient(self, grid):
        # print(grid.v.fourier_quads.shape)
        # print(self.avg_dist.shape)
        # spectrum = np.tensordot(grid.v.fourier_quads, self.avg_dist, axes=([1, 2], [0, 1]))
        # deriv = 1j * grid.v.modes * spectrum
        # return np.sum(deriv[:, None, None] *
        #               np.exp(1j * grid.v.modes[:, None, None] * grid.v.arr[None, :, :]), axis=0)
        return np.tensordot(self.avg_dist,
                            grid.v.local_basis.derivative_matrix, axes=([1], [0])) * grid.v.J[:, None].get()

    def field_particle_covariance(self, Elliptic, Grid):
        fluctuation_field = cp.array(self.delta_f * Elliptic.field.arr_nodal.get()[:, None, None])
        return trapz2(fluctuation_field, Grid.x.dx).get() / Grid.x.length

    def variance_of_field_particle_covariance(self, Elliptic, Grid, covariance):
        fluctuation_field = cp.array(self.delta_f * Elliptic.field.arr_nodal.get()[:, None, None])
        return trapz2((fluctuation_field - cp.array(covariance))**2, Grid.x.dx).get() / Grid.x.length

    # def compute_field_particle_covariance(self):
    #
    def total_density(self, grid):
        self.inverse_fourier_transform()
        self.compute_zero_moment(grid=grid)
        return self.zero_moment.integrate(grid=grid)

    def grid_flatten(self):
        return self.arr_nodal.reshape(self.x_res, self.v_res * self.order)

    def spectral_flatten(self):
        return self.arr.reshape(self.arr.shape[0], self.v_res * self.order)

    def initialize_bump_on_tail(self, grid, vt, u, chi, vb, vtb, perturbation=True):
        ix, iv = cp.ones_like(grid.x.device_arr), cp.ones_like(grid.v.device_arr)
        maxwellian = cp.tensordot(ix, grid.v.compute_maxwellian(thermal_velocity=vt,
                                                                drift_velocity=u), axes=0)
        bump = chi * cp.tensordot(ix, grid.v.compute_maxwellian(thermal_velocity=vtb,
                                                                drift_velocity=vb), axes=0)
        self.arr_nodal = (maxwellian + bump) / (1 + chi)
        self.fourier_transform()

        # compute perturbation
        if perturbation:
            # obtain eigenvalues by solving the dispersion relation
            sols = np.zeros_like(grid.x.wavenumbers) + 0j
            guess_r, guess_i = 0.03 / grid.x.fundamental, -0.003 / grid.x.fundamental
            for idx, wave in enumerate(grid.x.wavenumbers):
                if idx == 0:
                    continue
                solution = opt.root(dispersion.dispersion_fsolve, x0=np.array([guess_r, guess_i]),
                                    args=(wave, u, vt, chi, vb, vtb), jac=dispersion.jacobian_fsolve, tol=1.0e-15)
                guess_r, guess_i = solution.x
                sols[idx] = (guess_r + 1j * guess_i)

            plt.figure()
            plt.plot(grid.x.wavenumbers[1:], np.real(sols[1:]), 'ro--', label='Real part')
            plt.plot(grid.x.wavenumbers[1:], 20 * np.imag(sols[1:]), 'go--', label=r'Imaginary part, $\times 20$')
            plt.xlabel(r'Wavenumber $k\lambda_D$'), plt.ylabel(r'Phase velocity $\zeta/v_t$')
            plt.grid(True), plt.legend(loc='best'), plt.tight_layout()
            plt.show()

            # Build eigenfunction
            # df = cp.tensordot(ix,
            #                   grid.v.compute_maxwellian_gradient(thermal_velocity=vt, drift_velocity=u) +
            #                   chi * grid.v.compute_maxwellian_gradient(thermal_velocity=vtb, drift_velocity=vb),
            #                   axes=0) / (1 + chi)
            df = (grid.v.compute_maxwellian_gradient(thermal_velocity=vt, drift_velocity=u) +
                  chi * grid.v.compute_maxwellian_gradient(thermal_velocity=vtb, drift_velocity=vb)) / (1 + chi)

            # def eigenfunction(z, k):zs
            #     return (df / (z - grid.v.device_arr[None, :, :]) *
            #             cp.exp(1j * k * grid.x.device_arr[:, None, None])) / k
            def eigenfunction(z, k):
                pi2 = 2.0 * np.pi
                return (df / (z - grid.v.device_arr)) / k * cp.exp(1j * pi2 * cp.random.random(1))

            unstable_modes = grid.x.wavenumbers[np.imag(sols) > 0.003]
            mode_idxs = grid.x.device_modes[np.imag(sols) > 0.003]
            unstable_eigs = sols[np.imag(sols) > 0.003]
            # eig_sum, pi2 = 0, 2 * np.pi
            f1 = cp.zeros_like(self.arr) + 0j
            for idx in range(unstable_modes.shape[0]):
                f1[mode_idxs[idx], :, :] = -self.charge_mass * eigenfunction(unstable_eigs[idx], unstable_modes[idx])

        else:
            f1 = 0

        self.arr_nodal += 1.0e-3 * cp.fft.irfft(f1, axis=0, norm='forward')
        print('Finished initialization...')

    def fourier_transform(self):
        # self.arr = cp.fft.fftshift(cp.fft.fft(self.arr_nodal, axis=0, norm='forward'), axes=0)
        self.arr = cp.fft.rfft(self.arr_nodal, axis=0, norm='forward')

    def inverse_fourier_transform(self):
        # self.arr_nodal = cp.real(cp.fft.ifft(cp.fft.fftshift(self.arr, axes=0), norm='forward', axis=0))
        self.arr_nodal = cp.fft.irfft(self.arr, axis=0, norm='forward')


def trapz(y, dx):
    """ Custom trapz routine using cupy """
    return cp.sum(y[:-1] + y[1:]) * dx / 2.0


def trapz2(y, dx):
    return cp.sum(y[:-1, :] + y[1:, :], axis=0) * dx / 2.0

# grid
#             om_r = np.linspace(-0.1, 0.1, num=500)
#             om_i = np.linspace(-0.1, 0.1, num=500)
#
#             k = grid.x.fundamental
#
#             zr = om_r / k
#             zi = om_i / k
#
#             z = np.tensordot(zr, np.ones_like(zi), axes=0) + 1.0j * np.tensordot(np.ones_like(zr), zi, axes=0)
#
#             X, Y = np.meshgrid(om_r, om_i, indexing='ij')
#
#             # eps = dispersion_function(k, z, mr, tr, e_d)
#             chi = 0.05
#             vb = 5
#             vtb = chi ** (1 / 3) * vb
#             eps = dispersion.dispersion_function(k, z, drift_one=0, vt_one=1, two_scale=chi, drift_two=vb, vt_two=vtb)
#             cb = np.linspace(-1, 1, num=100)
#
#             # plt.figure()
#             # plt.contourf(X, Y, np.real(eps), cb, extend='both')
#
#             plt.figure()
#             plt.contour(X, Y, np.real(eps), 0, colors='r')
#             plt.contour(X, Y, np.imag(eps), 0, colors='g')
#             plt.grid(True)
#             plt.show()
