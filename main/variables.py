import numpy as np
import cupy as cp
import tools.dispersion as dispersion
import scipy.optimize as opt
import cupyx.scipy.signal as sig
import matplotlib.pyplot as plt

cp.random.seed(1111)


class SpaceScalar:
    def __init__(self, resolution):
        self.res = resolution
        self.arr_nodal, self.arr_spectral = None, None

    def fourier_transform(self):
        self.arr_spectral = cp.fft.fftshift(cp.fft.fft(self.arr_nodal, norm='forward'))
        # self.arr_spectral = cp.fft.rfft(self.arr_nodal, norm='forward')

    def inverse_fourier_transform(self):
        self.arr_nodal = cp.fft.ifft(cp.fft.fftshift(self.arr_spectral), norm='forward')
        # self.arr_nodal = cp.fft.irfft(self.arr_spectral, norm='forward')

    def integrate(self, grid):
        arr_add = cp.append(self.arr_nodal, self.arr_nodal[0])
        # x_add = cp.append(grid.x.device_arr, grid.x.device_arr[-1] + grid.x.dx)
        return trapz(arr_add, grid.x.dx)

    def integrate_energy(self, grid):
        arr = 0.5 * cp.real(self.arr_nodal) ** 2.0 + 0.5 * cp.imag(self.arr_nodal) ** 2.0
        arr_add = cp.append(arr, arr[0])
        # x_add = cp.append(grid.x.device_arr, grid.x.device_arr[-1] + grid.x.dx)
        return trapz(arr_add, grid.x.dx)

    def compute_wigner_distribution(self, grid):
        # Zero-pad
        fourier_functions = (self.arr_spectral[None, :] *
                             cp.exp(1j * grid.x.device_wavenumbers[None, :] * grid.x.device_arr[:, None]))
        return sig.fftconvolve(cp.conj(fourier_functions), fourier_functions, mode='same', axes=1)
        # fft1 = cp.fft.fft(cp.fft.fftshift(cp.conj(fourier_functions), axes=1), axis=1, norm='forward')
        # fft2 = cp.fft.fft(cp.fft.fftshift(fourier_functions, axes=1), axis=1, norm='forward')
        # return cp.fft.fftshift(cp.fft.ifft(fft1*fft2, axis=1, norm='forward'), axes=1)


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
        self.second_moment.arr_nodal = grid.v.second_moment(function=cp.real(self.arr_nodal), idx=[1, 2])
        return 0.5 * self.second_moment.integrate(grid=grid)

    def total_density(self, grid):
        self.inverse_fourier_transform()
        self.compute_zero_moment(grid=grid)
        return cp.real(self.zero_moment.integrate(grid=grid))

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
            zr = np.linspace(-7, 7, num=500)
            zi = np.linspace(-7, 7, num=500)
            z = np.tensordot(zr, np.ones_like(zi), axes=0) + 1.0j * np.tensordot(np.ones_like(zr), zi, axes=0)

            eps = dispersion.dispersion_function(grid.x.wavenumbers[0], z, drift_one=u, vt_one=vt,
                                                 two_scale=chi, drift_two=vb, vt_two=vtb)
            X, Y = np.meshgrid(zr, zi, indexing='ij')
            plt.figure()
            plt.contour(X, Y, np.real(eps), 0, colors='r')
            plt.contour(X, Y, np.imag(eps), 0, colors='g')
            plt.grid(True)
            plt.show()

            sols = np.zeros_like(grid.x.wavenumbers) + 0j
            guess_r, guess_i = 1.16, -2.22
            for idx, wave in enumerate(grid.x.wavenumbers):
                if wave == 0:
                    sols[idx] = guess_r + 1j * guess_i
                    continue
                solution = opt.root(dispersion.dispersion_fsolve, x0=np.array([guess_r, guess_i]),
                                    args=(wave, u, vt, chi, vb, vtb), jac=dispersion.jacobian_fsolve, tol=1.0e-15)
                guess_r, guess_i = solution.x
                sols[idx] = (guess_r + 1j * guess_i)

            plt.figure()
            plt.plot(grid.x.wavenumbers[1:], np.real(sols[1:]), 'r', label='real')
            plt.plot(grid.x.wavenumbers[1:], np.imag(sols[1:]), 'g', label='imag')  # * grid.x.wavenumbers * (2 ** 0.5)
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

            # def eigenfunction(z, k):
            #     return (df / (z - grid.v.device_arr[None, :, :]) *
            #             cp.exp(1j * k * grid.x.device_arr[:, None, None])) / k
            def eigenfunction(z, k):
                pi2 = 2.0 * np.pi
                return (df / (z - grid.v.device_arr)) / k * cp.exp(1j * pi2 * cp.random.random(1))

            unstable_modes = grid.x.wavenumbers[(np.imag(sols) > 0.003) & (grid.x.wavenumbers > 0)]
            mode_idxs = grid.x.device_mode_idxs[(np.imag(sols) > 0.003) & (grid.x.wavenumbers > 0)]
            unstable_eigs = sols[(np.imag(sols) > 0.003) & (grid.x.wavenumbers > 0)]
            # eig_sum, pi2 = 0, 2 * np.pi
            f1 = cp.zeros_like(self.arr) + 0j
            for idx in range(unstable_modes.shape[0]):
                this_idx = mode_idxs[idx]
                f1[this_idx, :, :] = (-self.charge_mass *
                                 eigenfunction(unstable_eigs[idx], unstable_modes[idx]))

        else:
            f1 = 0

        # self.arr_nodal += 1.0e-2 * cp.fft.irfft(f1, axis=0, norm='forward')
        self.arr += 1.0e-2 * f1  # * cp.fft.ifft(cp.fft.fftshift(f1, axes=0), axis=0, norm='forward')
        self.inverse_fourier_transform()
        print('Finished initialization...')

    def fourier_transform(self):
        self.arr = cp.fft.fftshift(cp.fft.fft(self.arr_nodal, axis=0, norm='forward'), axes=0)
        # self.arr = cp.fft.rfft(self.arr_nodal, axis=0, norm='forward')

    def inverse_fourier_transform(self):
        self.arr_nodal = cp.fft.ifft(cp.fft.fftshift(self.arr, axes=0), norm='forward', axis=0)
        # self.arr_nodal = cp.fft.irfft(self.arr, axis=0, norm='forward')


def trapz(y, dx):
    """ Custom trapz routine using cupy """
    return cp.sum(y[:-1] + y[1:]) * dx / 2.0

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
