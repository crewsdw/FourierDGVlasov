import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# from mpl_toolkits.axes_grid1 import make_axes_locatable


class Plotter:
    def __init__(self, grid, colormap='RdPu'):
        self.colormap = colormap
        self.grid = grid
        self.length = grid.x.length
        # Build structured grid, nodal
        self.X, self.V = np.meshgrid(grid.x.arr.flatten(), grid.v.arr.flatten(), indexing='ij')
        self.x = grid.x.arr
        self.k = grid.x.wavenumbers / grid.x.fundamental
        # Build structured grid, global spectral
        self.FX, self.FV = np.meshgrid(grid.x.wavenumbers / grid.x.fundamental, grid.v.arr.flatten(),
                                     indexing='ij')

    def distribution_contourf(self, distribution, plot_spectrum=True, remove_average=False):
        if distribution.arr_nodal is None:
            distribution.inverse_fourier_transform()
        if distribution.arr is None:
            distribution.fourier_transform()
        if remove_average:
            distribution.arr[0, :] = 0
            distribution.inverse_fourier_transform()

        dr = np.real(distribution.grid_flatten().get())
        di = np.imag(distribution.grid_flatten().get())
        cbr = np.linspace(np.amin(dr), np.amax(dr), num=300)
        cbi = np.linspace(np.amin(di), np.amax(di), num=300)

        plt.figure()
        plt.contourf(self.X, self.V, dr, cbr, cmap=self.colormap)
        plt.xlabel('x'), plt.ylabel('v'), plt.colorbar(), plt.tight_layout()
        plt.figure()
        plt.contourf(self.X, self.V, di, cbi, cmap=self.colormap)
        plt.xlabel('x'), plt.ylabel('v'), plt.colorbar(), plt.tight_layout()

        if plot_spectrum:
            spectrum_to_plot = distribution.spectral_flatten()
            spectrum_to_plot[self.grid.x.zero_idx, :] = 0.0
            spectrum = np.log(1.0 + np.absolute(spectrum_to_plot.get()))
            cb_s = np.linspace(np.amin(spectrum), 0.5 * np.amax(spectrum), num=100)

            plt.figure()
            plt.contourf(self.FX, self.FV, spectrum, cb_s, extend='both')  # , cmap=self.colormap)
            plt.xlabel('mode'), plt.ylabel('v'), plt.colorbar(), plt.tight_layout()

    def wave_phase_space_contourf(self, field, grid):
        wigner = field.compute_wigner_distribution(grid=grid).get()
        # print(wigner[grid.x.zero_idx.get(), :])
        wig_r = np.real(wigner)
        wig_i = np.imag(wigner)

        cbr = np.linspace(np.amin(wig_r), np.amax(wig_r), num=100)
        cbi = np.linspace(np.amin(wig_i), np.amax(wig_i), num=100)

        plt.figure()
        plt.contourf(grid.x.X, grid.x.K, wig_r, cbr)
        plt.xlabel(r'Position $x$'), plt.ylabel(r'Wavenumber $k$')
        plt.colorbar(), plt.title('Real Wigner'), plt.tight_layout()

        plt.figure()
        plt.contourf(grid.x.X, grid.x.K, wig_i, cbi)
        plt.xlabel(r'Position $x$'), plt.ylabel(r'Wavenumber $k$')
        plt.colorbar(), plt.title('Imag Wigner'), plt.tight_layout()

        plt.show()

    def spatial_scalar_plot(self, scalar, y_axis, spectrum=True, quadratic=False):
        if scalar.arr_nodal is None:
            scalar.inverse_fourier_transform()

        plt.figure()
        plt.plot(self.x.flatten(), np.real(scalar.arr_nodal.flatten().get()), 'o', label='real')
        plt.plot(self.x.flatten(), np.imag(scalar.arr_nodal.flatten().get()), 'o', label='imag')
        plt.xlabel('x'), plt.ylabel(y_axis), plt.legend(loc='best')
        plt.grid(True), plt.tight_layout()

        if spectrum:
            plt.figure()
            spectral_arr = scalar.arr_spectral.flatten().get()
            if not quadratic:
                plt.plot(self.k.flatten(), np.real(spectral_arr), 'ro', label='real')
                plt.plot(self.k.flatten(), np.imag(spectral_arr), 'go', label='imaginary')
                plt.legend(loc='best')
            if quadratic:
                plt.plot(self.k.flatten(), np.absolute(spectral_arr)**2.0, 'o')
            plt.xlabel('Modes'), plt.ylabel(y_axis + ' spectrum')
            plt.grid(True), plt.tight_layout()

    def time_series_plot(self, time_in, series_in, y_axis, log=False, give_rate=False):
        time, series = time_in, np.absolute(series_in.get()) / self.length
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

    def animate_line_plot(self, saved_array):
        fig, ax = plt.subplots()
        # ax.plot(self.x, saved_array[0, :], 'o--')
        # plt.show()
        plot, = ax.plot(self.x.flatten(), saved_array[0, :], 'o--')
        plt.grid(True)
        ax.set_ylim([0.995, 1.005])

        def animate_frame(idx):
            # clear existing contours
            # ax.collections = []
            # ax.patches = []
            # ax.clear()
            plot.set_data(self.x, saved_array[idx, :])

            # plot line-plot
            # ax.plot(self.x.flatten(), saved_array[idx, :], 'o--')

        anim = animation.FuncAnimation(fig, animate_frame, frames=len(saved_array))
        anim.save(filename='animation.mp4')

    def show(self):
        plt.show()
