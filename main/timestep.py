import numpy as np
# import scipy.integrate as spint
import time as timer
import fluxes as fx
import variables as var
import matplotlib.pyplot as plt
import cupy as cp

nonlinear_ssp_rk_switch = {
    2: [[1 / 2, 1 / 2, 1 / 2]],
    3: [[3 / 4, 1 / 4, 1 / 4],
        [1 / 3, 2 / 3, 2 / 3]]
}


class StepperSingleSpecies:
    def __init__(self, dt, step, resolutions, order, steps, grid):
        self.x_res, self.v_res = resolutions
        self.resolutions = resolutions
        self.order = order
        self.dt = dt
        self.step = step
        self.steps = steps
        self.flux = fx.DGFlux(resolutions=resolutions, x_res=grid.x.device_wavenumbers.shape[0],
                              order=order, charge_mass=-1.0)
        self.flux.initialize_zero_pad(grid=grid)

        # RK coefficients
        self.rk_coefficients = np.array(nonlinear_ssp_rk_switch.get(3, "nothing"))

        # tracking arrays
        self.time = 0
        self.next_time = 0
        self.field_energy = np.array([])
        self.time_array = np.array([])
        self.thermal_energy = np.array([])
        self.density_array = np.array([])
        # self.saved_field = np.array([])
        num = int(self.steps // 20 + 1)

    def main_loop_adams_bashforth(self, distribution, elliptic, grid):  # , plotter, plot=True):
        """
            Evolve the Vlasov equation in wavenumber space using the Adams-Bashforth time integration scheme
        """
        print('Beginning main loop')

        # Compute first two steps with ssp-rk3 and save fluxes
        # zeroth step
        elliptic.poisson_solve_single_species(distribution=distribution, grid=grid)
        self.flux.semi_discrete_rhs(distribution=distribution, elliptic=elliptic, grid=grid)
        flux0 = self.flux.output.arr

        # first step
        self.ssp_rk3(distribution=distribution, elliptic=elliptic, grid=grid)
        self.time += self.dt
        # save fluxes
        elliptic.poisson_solve_single_species(distribution=distribution, grid=grid)
        self.flux.semi_discrete_rhs(distribution=distribution, elliptic=elliptic, grid=grid)
        flux1 = self.flux.output.arr

        # store first two fluxes
        previous_fluxes = [flux1, flux0]

        # Begin loop
        for i in range(1, self.steps):
            previous_fluxes = self.adams_bashforth(distribution=distribution,
                                                   elliptic=elliptic, grid=grid, prev_fluxes=previous_fluxes)
            self.time += self.step

            # Check out noise levels. Note: Incorrect to naively adapt time-step for adams-bashforth method
            # plt.figure()
            # plt.plot(grid.x.wavenumbers[1:], np.absolute(distribution.zero_moment.arr_spectral.get())[1:], 'o')
            # plt.show()
            # largest_idx = cp.amax(grid.x.device_modes[cp.absolute(distribution.zero_moment.arr_spectral) > 1.0e-9])
            # # adapt time-step
            # # print(self.dt)
            # self.dt = 0.05 / (grid.x.device_wavenumbers[largest_idx] * grid.v.high)
            # self.step = self.dt.get()

            if i % 20 == 0:
                self.time_array = np.append(self.time_array, self.time)
                elliptic.poisson_solve_single_species(distribution=distribution, grid=grid)
                self.field_energy = np.append(self.field_energy, elliptic.compute_field_energy(grid=grid))
                self.thermal_energy = np.append(self.thermal_energy,
                                                distribution.total_thermal_energy(grid=grid))
                self.density_array = np.append(self.density_array,
                                               distribution.total_density(grid=grid))
                print('Took step, time is {:0.3e}'.format(self.time))

        return distribution

    def ssp_rk3(self, distribution, elliptic, grid):
        # Stage set-up
        stage0 = var.Distribution(resolutions=self.resolutions, order=self.order, charge_mass=None)
        stage1 = var.Distribution(resolutions=self.resolutions, order=self.order, charge_mass=None)

        # zero stage
        elliptic.poisson_solve_single_species(distribution=distribution, grid=grid)
        self.flux.semi_discrete_rhs(distribution=distribution, elliptic=elliptic, grid=grid)
        stage0.arr = distribution.arr + self.dt * self.flux.output.arr

        # first stage
        elliptic.poisson_solve_single_species(distribution=stage0, grid=grid)
        self.flux.semi_discrete_rhs(distribution=stage0, elliptic=elliptic, grid=grid)
        stage1.arr = (
                self.rk_coefficients[0, 0] * distribution.arr +
                self.rk_coefficients[0, 1] * stage0.arr +
                self.rk_coefficients[0, 2] * self.dt * self.flux.output.arr
        )

        # second stage
        elliptic.poisson_solve_single_species(distribution=stage1, grid=grid)
        self.flux.semi_discrete_rhs(distribution=stage1, elliptic=elliptic, grid=grid)
        distribution.arr = (
                self.rk_coefficients[1, 0] * distribution.arr +
                self.rk_coefficients[1, 1] * stage1.arr +
                self.rk_coefficients[1, 2] * self.dt * self.flux.output.arr
        )

    def adams_bashforth(self, distribution, elliptic, grid, prev_fluxes):
        # Compute flux at this point
        elliptic.poisson_solve_single_species(distribution=distribution, grid=grid)
        self.flux.semi_discrete_rhs(distribution=distribution, elliptic=elliptic, grid=grid)

        # Update distribution
        distribution.arr += self.dt * (23 / 12 * self.flux.output.arr -
                                       4 / 3 * prev_fluxes[0] +
                                       5 / 12 * prev_fluxes[1])
        # update previous_fluxes (n-2, n-1)
        return [self.flux.output.arr, prev_fluxes[0]]


class StepperTwoSpecies:
    def __init__(self, dt, step, resolutions, order, steps, grids, charge_mass):
        self.x_res, self.v_res = resolutions
        self.resolutions = resolutions
        self.order = order
        self.dt = dt
        self.step = step
        self.steps = steps
        self.flux_e = fx.DGFlux(resolutions=resolutions, order=order, charge_mass=-1.0)
        self.flux_e.initialize_zero_pad(grid=grids[0])
        self.flux_p = fx.DGFlux(resolutions=resolutions, order=order, charge_mass=+1.0 / charge_mass)
        self.flux_p.initialize_zero_pad(grid=grids[1])

        self.mass_ratio = charge_mass

        # RK coefficients
        self.rk_coefficients = np.array(nonlinear_ssp_rk_switch.get(3, "nothing"))

        # tracking arrays
        self.time = 0
        self.next_time = 0
        self.field_energy = np.array([])
        self.time_array = np.array([])
        self.thermal_energy_e = np.array([])
        self.thermal_energy_p = np.array([])
        self.density_array_e = np.array([])
        self.density_array_p = np.array([])
        num = int(self.steps // 20 + 1)
        # self.saved_density = np.zeros((num, self.x_res))

    def main_loop_ssprk3(self, distribution_e, distribution_p, elliptic, grids):  # , plotter, plot=True):
        # just SSP-RK3
        print('Beginning main loop')
        for i in range(self.steps):
            self.next_time = self.time + self.step
            span = [self.time, self.next_time]
            self.ssp_rk3(distribution_e=distribution_e, distribution_p=distribution_p, elliptic=elliptic, grids=grids)
            self.time += self.step
            if i % 10 == 0:
                self.time_array = np.append(self.time_array, self.time)
                elliptic.poisson_solve_two_species(distribution_e=distribution_e,
                                                   distribution_p=distribution_p, grids=grids)
                self.field_energy = np.append(self.field_energy, elliptic.compute_field_energy(grid=grids[0]))
                self.thermal_energy_e = np.append(self.thermal_energy_e,
                                                  distribution_e.total_thermal_energy(grid=grids[0]))
                self.thermal_energy_p = np.append(self.thermal_energy_p,
                                                  self.mass_ratio * distribution_p.total_thermal_energy(grid=grids[1]))
                self.density_array_e = np.append(self.density_array_e,
                                                 distribution_e.total_density(grid=grids[0]))
                self.density_array_p = np.append(self.density_array_p,
                                                 distribution_p.total_density(grid=grids[1]))

                print('Took step, time is {:0.3e}'.format(self.time))

        return distribution_e, distribution_p

    def main_loop_adams_bashforth(self, distribution_e, distribution_p, elliptic, grids):  # , plotter, plot=True):
        # using adams-bashforth method
        print('Beginning main loop')

        # Compute first two steps with ssp-rk3 and save fluxes
        # zeroth step
        elliptic.poisson_solve_two_species(distribution_e=distribution_e, distribution_p=distribution_p, grids=grids)
        self.flux_e.semi_discrete_rhs(distribution=distribution_e, elliptic=elliptic, grid=grids[0])
        self.flux_p.semi_discrete_rhs(distribution=distribution_p, elliptic=elliptic, grid=grids[1])
        flux0e, flux0p = self.flux_e.output.arr, self.flux_p.output.arr

        # first step
        self.ssp_rk3(distribution_e=distribution_e, distribution_p=distribution_p, elliptic=elliptic, grids=grids)
        self.time += self.dt
        # save fluxes
        elliptic.poisson_solve_two_species(distribution_e=distribution_e, distribution_p=distribution_p, grids=grids)
        self.flux_e.semi_discrete_rhs(distribution=distribution_e, elliptic=elliptic, grid=grids[0])
        self.flux_p.semi_discrete_rhs(distribution=distribution_p, elliptic=elliptic, grid=grids[1])
        flux1e, flux1p = self.flux_e.output.arr, self.flux_p.output.arr

        # store first two fluxes
        previous_fluxes = [[flux1e, flux0e], [flux1p, flux0p]]
        # self.saved_density[0, :] = distribution_e.zero_moment.arr_nodal.get()
        # Begin loop
        for i in range(1, self.steps):
            previous_fluxes = self.adams_bashforth(distribution_e=distribution_e, distribution_p=distribution_p,
                                                   elliptic=elliptic, grids=grids, prev_fluxes=previous_fluxes)
            self.time += self.step

            if i % 20 == 0:
                self.time_array = np.append(self.time_array, self.time)
                elliptic.poisson_solve_two_species(distribution_e=distribution_e,
                                                   distribution_p=distribution_p, grids=grids)
                self.field_energy = np.append(self.field_energy, elliptic.compute_field_energy(grid=grids[0]))
                self.thermal_energy_e = np.append(self.thermal_energy_e,
                                                  distribution_e.total_thermal_energy(grid=grids[0]))
                self.thermal_energy_p = np.append(self.thermal_energy_p,
                                                  self.mass_ratio * distribution_p.total_thermal_energy(grid=grids[1]))
                self.density_array_e = np.append(self.density_array_e,
                                                 distribution_e.total_density(grid=grids[0]))
                self.density_array_p = np.append(self.density_array_p,
                                                 distribution_p.total_density(grid=grids[1]))
                # self.saved_density[i // 20, :] = distribution_e.zero_moment.arr_nodal.get()

                # print(self.saved_density.shape)
                # quit()
                print('Took step, time is {:0.3e}'.format(self.time))

        return distribution_e, distribution_p

    def ssp_rk3(self, distribution_e, distribution_p, elliptic, grids):
        # Stage set-up
        stage0_e = var.Distribution(resolutions=self.resolutions, order=self.order, charge_mass=None)
        stage0_p = var.Distribution(resolutions=self.resolutions, order=self.order, charge_mass=None)

        stage1_e = var.Distribution(resolutions=self.resolutions, order=self.order, charge_mass=None)
        stage1_p = var.Distribution(resolutions=self.resolutions, order=self.order, charge_mass=None)

        # zero stage
        elliptic.poisson_solve_two_species(distribution_e=distribution_e, distribution_p=distribution_p, grids=grids)
        self.flux_e.semi_discrete_rhs(distribution=distribution_e, elliptic=elliptic, grid=grids[0])
        self.flux_p.semi_discrete_rhs(distribution=distribution_p, elliptic=elliptic, grid=grids[1])

        stage0_e.arr = distribution_e.arr + self.dt * self.flux_e.output.arr
        stage0_p.arr = distribution_p.arr + self.dt * self.flux_p.output.arr

        # first stage
        elliptic.poisson_solve_two_species(distribution_e=stage0_e, distribution_p=stage0_p, grids=grids)
        self.flux_e.semi_discrete_rhs(distribution=stage0_e, elliptic=elliptic, grid=grids[0])
        self.flux_p.semi_discrete_rhs(distribution=stage0_p, elliptic=elliptic, grid=grids[1])

        stage1_e.arr = (
                self.rk_coefficients[0, 0] * distribution_e.arr +
                self.rk_coefficients[0, 1] * stage0_e.arr +
                self.rk_coefficients[0, 2] * self.dt * self.flux_e.output.arr
        )
        stage1_p.arr = (
                self.rk_coefficients[0, 0] * distribution_p.arr +
                self.rk_coefficients[0, 1] * stage0_p.arr +
                self.rk_coefficients[0, 2] * self.dt * self.flux_p.output.arr
        )
        # second stage
        elliptic.poisson_solve_two_species(distribution_e=stage1_e, distribution_p=stage1_p, grids=grids)
        self.flux_e.semi_discrete_rhs(distribution=stage1_e, elliptic=elliptic, grid=grids[0])
        self.flux_p.semi_discrete_rhs(distribution=stage1_p, elliptic=elliptic, grid=grids[1])

        distribution_e.arr = (
                self.rk_coefficients[1, 0] * distribution_e.arr +
                self.rk_coefficients[1, 1] * stage1_e.arr +
                self.rk_coefficients[1, 2] * self.dt * self.flux_e.output.arr
        )
        distribution_p.arr = (
                self.rk_coefficients[1, 0] * distribution_p.arr +
                self.rk_coefficients[1, 1] * stage1_p.arr +
                self.rk_coefficients[1, 2] * self.dt * self.flux_p.output.arr
        )

    def adams_bashforth(self, distribution_e, distribution_p, elliptic, grids, prev_fluxes):
        # Compute flux at this point
        elliptic.poisson_solve_two_species(distribution_e=distribution_e, distribution_p=distribution_p, grids=grids)
        self.flux_e.semi_discrete_rhs(distribution=distribution_e, elliptic=elliptic, grid=grids[0])
        self.flux_p.semi_discrete_rhs(distribution=distribution_p, elliptic=elliptic, grid=grids[1])

        # Update distribution
        distribution_e.arr += self.dt * (23 / 12 * self.flux_e.output.arr -
                                         4 / 3 * prev_fluxes[0][0] +
                                         5 / 12 * prev_fluxes[0][1])
        distribution_p.arr += self.dt * (23 / 12 * self.flux_p.output.arr -
                                         4 / 3 * prev_fluxes[1][0] +
                                         5 / 12 * prev_fluxes[1][1])
        # update previous_fluxes
        return [[self.flux_e.output.arr, prev_fluxes[0][0]],
                [self.flux_p.output.arr, prev_fluxes[1][0]]]
