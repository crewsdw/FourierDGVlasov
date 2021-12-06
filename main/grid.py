import numpy as np
import cupy as cp
import basis as b
import tools.plasma_dispersion as pd
import matplotlib.pyplot as plt


class SpaceGrid:
    """ In this scheme, the spatial grid is uniform and transforms are accomplished by DFT """
    def __init__(self, low, high, elements):
        # grid limits and elements
        self.low, self.high = low, high
        self.elements = elements

        # domain and element widths
        self.length = self.high - self.low
        self.dx = self.length / self.elements

        # element Jacobian
        self.J = 2.0 / self.dx

        # arrays
        self.arr, self.device_arr = None, None
        self.create_grid()

        # spectral properties
        self.modes = elements // 2 + 1  # Nyquist frequency
        self.fundamental = 2.0 * np.pi / self.length
        # self.wavenumbers = self.fundamental * np.arange(-self.modes, self.modes)
        self.wavenumbers = self.fundamental * np.arange(self.modes)
        # print(self.wavenumbers)
        self.device_modes = cp.arange(self.modes)
        self.device_wavenumbers = cp.array(self.wavenumbers)
        self.zero_idx = 0  # int(self.modes)
        # self.two_thirds_low = int((1 * self.modes)//3 + 1)
        # self.two_thirds_high = self.wavenumbers.shape[0] - self.two_thirds_low
        self.pad_width = int((1 * self.modes)//3 + 1)
        # print(self.two_thirds_low)
        # print(self.two_thirds_high)
        print(self.length)
        print(self.fundamental)

    def create_grid(self):
        """ Build evenly spaced grid, assumed periodic """
        self.arr = np.linspace(self.low, self.high - self.dx, num=self.elements)
        self.device_arr = cp.asarray(self.arr)


class VelocityGrid:
    """ In this experiment, the velocity grid is an LGL quadrature grid """
    def __init__(self, low, high, elements, order):
        self.low, self.high = low, high
        self.elements, self.order = elements, order
        self.local_basis = b.LGLBasis1D(order=self.order)
        # self.local_basis = b.GLBasis1D(order=self.order)

        # domain and element widths
        self.length = self.high - self.low
        self.dx = self.length / self.elements

        # arrays
        self.arr, self.device_arr = None, None
        self.mid_points = None
        self.create_even_grid()

        # stretch / transform elements
        self.pole_distance = 5
        self.dx_grid = None
        self.stretch_grid()
        self.modes = 2.0 * np.pi * np.arange(1-int(2*self.elements), int(2*self.elements)) / self.length

        # jacobian
        # self.J = 2.0 / self.dx
        self.J = cp.asarray(2.0 / self.dx_grid)
        # plt.figure()
        # x = np.linspace(-500, 500, num=5)
        # X, V = np.meshgrid(x, self.arr.flatten(), indexing='ij')
        # plt.plot(X, V, 'ko--')
        # plt.plot(X.T, V.T, 'ko--')
        # for i in range(self.elements):
        #     plt.plot(np.zeros_like(self.arr[i, :]), self.arr[i, :], 'ko')
        # plt.show()

        # global quad weights
        self.global_quads = cp.tensordot(cp.ones(elements),
                                         cp.asarray(self.local_basis.weights), axes=0)

        # global translation matrix
        mid_identity = np.tensordot(self.mid_points, np.eye(self.local_basis.order), axes=0)
        self.translation_matrix = cp.asarray(mid_identity +
                                             self.local_basis.translation_matrix[None, :, :] /
                                             self.J[:, None, None].get())

        # quad matrix
        self.fourier_quads = (self.local_basis.weights[None, None, :] *
                              np.exp(-1j*self.modes[:, None, None]*self.arr[None, :, :]) /
                              self.J[None, :, None].get()) / self.length

    def create_even_grid(self):
        """ Build global grid """
        # translate to [0, 1]
        nodes_iso = (np.array(self.local_basis.nodes) + 1) / 2
        # element left boundaries (including ghost elements)
        xl = np.linspace(self.low, self.high - self.dx, num=self.elements)
        # construct coordinates
        self.arr = np.zeros((self.elements, self.order))
        for i in range(self.elements):
            self.arr[i, :] = xl[i] + self.dx * np.array(nodes_iso)
        # send to device
        self.device_arr = cp.asarray(self.arr)
        self.mid_points = np.array([0.5 * (self.arr[i, -1] + self.arr[i, 0]) for i in range(self.elements)])

    def stretch_grid(self):
        # Investigate grid mapping
        alphas, betas = (np.array([0.6, 0.6, 0.6, 0.6]),  # 0.64]),  # , 0.705, 0.705, 0.705, 0.705]),
                         np.array([0.5, 0.5, 0.5, 0.5]))
        # alphas, betas = (np.array([0.62, 0.62, 0.62, 0.62]),  # , 0.705, 0.705, 0.705, 0.705]),
        #                  np.array([0.5, 0.5, 0.5, 0.5]))  # , 0.6, 0.6, 0.6, 0.6]))
        plt.figure()
        plt.plot(self.arr.flatten(), self.iterate_map_asym(self.arr, alphas[:-2], betas[:-2]).flatten(),
                 'k', label=r'Iteration 1')
        plt.plot(self.arr.flatten(), self.iterate_map_asym(self.arr, alphas[:-1], betas[:-1]).flatten(),
                 'k', label=r'Iteration 2')
        plt.plot(self.arr.flatten(), self.iterate_map_asym(self.arr, alphas, betas).flatten(),
                 'r', label=r'Iteration 3')
        # plt.plot(self.arr.flatten(), self.grid_map_asym(self.arr, alpha=0.25, beta=0.5).flatten(),
        #          'b', label=r'$\alpha=0.25, \beta=0.5$')
        # plt.plot(self.arr.flatten(), self.grid_map_asym(self.arr, alpha=0.5, beta=0.25).flatten(),
        #          'g', label=r'$\alpha=0.5, \beta=0.25$')
        # plt.plot(self.arr.flatten(), self.iterate_map(self.arr, orders=[0.5]).flatten(),
        #          'k', label='Iteration 1')
        # plt.plot(self.arr.flatten(), self.iterate_map(self.arr, orders=[0.5, 0.5]).flatten(),
        #          'g', label='Iteration 2')
        # plt.plot(self.arr.flatten(), self.iterate_map(self.arr, orders=[0.5, 0.5, 0.9]).flatten(),
        #          'r', label='Iteration 3')
        # plt.plot(self.arr.flatten(), self.iterate_map(self.arr, orders=[0.5, 0.5, 0.9, 0.9, 0.9, 0.9]).flatten(),
        #          'b', label='Iteration 4')
        # plt.plot(self.arr.flatten(), self.iterate_map(self.arr, orders=[0.25, 0.5, 0.8, 0.8, 0.8]).flatten(),
        #          'k', label='Iteration 5')
        # plt.plot(self.arr.flatten(), self.iterate_map(self.arr, orders=[0.25, 0.5, 0.8, 0.8, 0.8, 0.8]).flatten(),
        #          'k', label='Iteration 6')
        plt.xlabel('Input points'), plt.ylabel('Output points')
        plt.grid(True), plt.legend(loc='best'), plt.tight_layout()
        plt.show()
        # Map points
        # Map lows and highs
        # orders = [0.4, 0.5, 0.8, 0.8, 0.8]  # , 0.8, 0.8]
        # mapped_lows = self.iterate_map(self.arr[:, 0], orders=orders)
        # mapped_highs = self.iterate_map(self.arr[:, -1], orders=orders)
        # alphas, betas = np.array([0.3, 0.55]), np.array([0.35, 0.6])
        mapped_lows = self.iterate_map_asym(self.arr[:, 0], alphas=alphas, betas=betas)
        mapped_highs = self.iterate_map_asym(self.arr[:, -1], alphas=alphas, betas=betas)
        self.dx_grid = mapped_highs - mapped_lows
        # self.dx_grid = self.dx * self.grid_map(self.mid_points)  # mapped_highs - mapped_lows
        # print(self.dx_grid)
        # xl = np.linspace(self.low, self.high - self.dx, num=self.elements)
        # Overwrite coordinate array
        nodes_iso = (np.array(self.local_basis.nodes) + 1) / 2
        lows = np.zeros(self.elements+1)
        lows[0] = self.low
        for i in range(self.elements):
            self.arr[i, :] = lows[i] + self.dx_grid[i] * np.array(nodes_iso)
            lows[i+1] = self.arr[i, -1]
        # send to device
        self.device_arr = cp.asarray(self.arr)
        self.mid_points = np.array([0.5 * (self.arr[i, -1] + self.arr[i, 0]) for i in range(self.elements)])

    def iterate_map(self, points, orders):
        for order in orders:
            points = self.grid_map(points, order)
        return points

    def iterate_map_asym(self, points, alphas, betas):
        for idx, alpha in enumerate(alphas):
            points = self.grid_map_asym(points, alpha, betas[idx])
        return points

    def grid_map_asym(self, points, alpha, beta):
        return (self.low * ((self.high - points) / self.length) ** alpha +
                self.high * ((points - self.low) / self.length) ** beta)

    def grid_map(self, points, order):
        return (self.low * ((self.high - points) / self.length) ** order +
                self.high * ((points - self.low) / self.length) ** order)
        # return (self.low * ((points - self.high) / self.length) ** order * ((points-5) / (self.low-5)) +
        #         5 * ((points-self.high) / (self.high - 5)) ** order * ((points-self.low) / (self.low - 5)) ** order +
        #         self.high * ((points - self.low) / self.length) ** order * ((points-5) / (self.high-5)))
        # return self.pole_distance * ((self.length + self.pole_distance) / (
        #         (points - (self.high + self.pole_distance)) *
        #         (self.low - self.pole_distance - points)))

    # def grid_map_jacobian(self, points, order):
    #     return self.grid_map(points, order) * (1 / (points - (self.low - self.pole_distance)) +
    #                                     1 / (points - (self.high + self.pole_distance)))

    def zero_moment(self, function, idx):
        return cp.tensordot(self.global_quads / self.J[:, None], function, axes=([0, 1], idx))  # / self.J

    def second_moment(self, function, idx):
        return cp.tensordot(self.global_quads / self.J[:, None], cp.multiply(self.device_arr[None, :, :] ** 2.0,
                                                           function),
                            axes=([0, 1], idx))

    def compute_maxwellian(self, thermal_velocity, drift_velocity):
        return cp.exp(-0.5 * ((self.device_arr - drift_velocity) /
                        thermal_velocity) ** 2.0) / (np.sqrt(2.0 * np.pi) * thermal_velocity)

    def compute_maxwellian_gradient(self, thermal_velocity, drift_velocity):
        return (-1.0*((self.device_arr - drift_velocity) / thermal_velocity ** 2.0) *
                self.compute_maxwellian(thermal_velocity=thermal_velocity, drift_velocity=drift_velocity))


class PhaseSpace:
    """ In this experiment, PhaseSpace consists of equispaced nodes and a finite LGL quad grid in velocity"""
    def __init__(self, lows, highs, elements, order):
        self.x = SpaceGrid(low=lows[0], high=highs[0], elements=elements[0])
        self.v = VelocityGrid(low=lows[1], high=highs[1], elements=elements[1], order=order)

    # def fourier_transform(self, function):
    #     return np.fft.fft(function, axis=0)

    def eigenfunction(self, thermal_velocity, drift_velocity, eigenvalue, beams='two-stream'):
        if beams == 'one':
            df = self.v.compute_maxwellian_gradient(thermal_velocity=thermal_velocity,
                                                    drift_velocity=drift_velocity)
            zeta = eigenvalue / self.x.fundamental
            denom = zeta - self.v.device_arr
            denom_abs = np.absolute(denom)
            denom_ang = np.angle(denom)
            # denom = np.absolute(zeta - self.v.device_arr)
            # v_part = df / (eigenvalue - self.x.fundamental * self.v.device_arr) / self.x.fundamental
            v_part = df / denom_abs / (self.x.fundamental ** 2.0)  # * np.exp(-1j * denom_ang)
            xv_part = cp.cos(self.x.fundamental * self.x.device_arr[:, None, None] - denom_ang[None, :, :])
            return xv_part * v_part[None, :, :]
            # v_part = df / (zeta - self.v.device_arr) / (self.x.fundamental ** 2.0)
            # z = eigenvalue / (0.5 * np.sqrt(2))
            # eps = 0  # 1.0 - 0.5 * pd.Zprime(z) / (self.x.fundamental ** 2.0)
            # eps_prime = -0.5 * pd.Zdoubleprime(z) / (self.x.fundamental ** 3.0)
            # v_part = df / (eps + (eigenvalue - self.x.fundamental * self.v.device_arr) * eps_prime)
            # return cp.tensordot(cp.cos(self.x.fundamental * self.x.device_arr), v_part, axes=0)
            # return cp.tensordot(cp.exp(1j * self.x.fundamental * self.x.device_arr), v_part, axes=0)
            # return cp.tensordot(cp.ones_like(self.x.device_arr), v_part, axes=0)

        if beams == 'two-stream':
            df1 = self.v.compute_maxwellian_gradient(thermal_velocity=thermal_velocity,
                                                     drift_velocity=drift_velocity[0])
            df2 = self.v.compute_maxwellian_gradient(thermal_velocity=thermal_velocity,
                                                     drift_velocity=drift_velocity[1])
            df = 0.5 * (df1 + df2)
            v_part = cp.divide(df, self.v.device_arr - eigenvalue)
            return cp.tensordot(cp.exp(1j * self.x.fundamental * self.x.device_arr), v_part, axes=0) * cp.exp(0.1j)
