import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from mpl_toolkits.axes_grid1 import make_axes_locatable


class Plotter:
    def __init__(self, grid, colormap='RdPu'):
        self.colormap = colormap
        self.grid = grid
        # Build structured grid, nodal
        self.X, self.V = np.meshgrid(grid.x.arr.flatten(), grid.v.arr.flatten(), indexing='ij')
        self.x = grid.x.arr
        self.k = grid.x.wavenumbers / grid.x.fundamental
        # Build structured grid, global spectral
        self.FX, self.FV = np.meshgrid(grid.x.wavenumbers / grid.x.fundamental, grid.v.arr.flatten(),
                                     indexing='ij')

    def distribution_contourf(self, distribution, plot_spectrum=True):
        if distribution.arr_nodal is None:
            distribution.inverse_fourier_transform()
        if distribution.arr is None:
            distribution.fourier_transform()

        # spectrum = np.log(1.0 + np.absolute(distribution.spectral_flatten().get()))
        # cb = np.linspace(np.amin(distribution.arr_nodal), np.amax(distribution.arr_nodal), num=100).get()
        # Compute average distribution: trapz integration
        # avg_dist = cp.sum(distribution.arr[:-1, :, :] + distribution.arr[1:, :, :], axis=0) * self.grid.v.dx / 2.0
        # avg_dist = distribution.arr[self.grid.x.zero_idx]
        # spectrum_to_plot = (distribution.arr - avg_dist[None, :, :]).reshape(self.grid.x.elements,
        #                                                                      self.grid.v.elements * self.grid.v.order)
        spectrum_to_plot = distribution.spectral_flatten()
        spectrum_to_plot[self.grid.x.zero_idx, :] = 0.0
        # spectrum = np.log(1.0 + np.absolute(distribution.spectral_flatten().get()))
        spectrum = np.log(1.0 + np.absolute(spectrum_to_plot.get()))
        cb = np.linspace(np.amin(distribution.arr_nodal.get()), np.amax(distribution.arr_nodal.get()), num=100)  # 0
        cb_s = np.linspace(np.amin(spectrum), np.amax(spectrum), num=100)

        plt.figure()
        plt.contourf(self.X, self.V, distribution.grid_flatten().get(), cb, cmap=self.colormap)
        plt.xlabel('x'), plt.ylabel('v'), plt.colorbar(), plt.tight_layout()

        if plot_spectrum:
            plt.figure()
            plt.contourf(self.FX, self.FV, spectrum, cb_s)  # , cmap=self.colormap)
            plt.xlabel('mode'), plt.ylabel('v'), plt.colorbar(), plt.tight_layout()

    def spatial_scalar_plot(self, scalar, y_axis, spectrum=True):
        if scalar.arr_nodal is None:
            scalar.inverse_fourier_transform()

        plt.figure()
        plt.plot(self.x.flatten(), scalar.arr_nodal.flatten().get(), 'o')
        plt.xlabel('x'), plt.ylabel(y_axis)
        plt.grid(True), plt.tight_layout()

        if spectrum:
            plt.figure()
            spectrum = scalar.arr_spectral.flatten().get()
            plt.plot(self.k.flatten(), np.real(spectrum), 'ro', label='real')
            plt.plot(self.k.flatten(), np.imag(spectrum), 'go', label='imaginary')
            plt.xlabel('Modes'), plt.ylabel(y_axis + ' spectrum')
            plt.grid(True), plt.legend(loc='best'), plt.tight_layout()

    def time_series_plot(self, time_in, series_in, y_axis, log=False, give_rate=False):
        time, series = time_in, series_in.get()
        plt.figure()
        if log:
            plt.semilogy(time, series, 'o--')
        else:
            plt.plot(time, series, 'o--')
        plt.xlabel('Time')
        plt.ylabel(y_axis)
        plt.grid(True), plt.tight_layout()
        if give_rate:
            lin_fit = np.polyfit(time, np.log(series), 1)
            print('Numerical rate: {:0.10e}'.format(lin_fit[0]))
            print('cf. exact rate: {:0.10e}'.format(2 * 2.409497728e-01))
            print('The difference is {:0.10e}'.format(lin_fit[0] - 2 * 2.409497728e-01))

    def show(self):
        plt.show()
