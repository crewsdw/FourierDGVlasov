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
    def __init__(self, dt, step, resolutions, order, steps, grid, nu):
        self.x_res, self.v_res = resolutions
        self.resolutions = resolutions
        self.order = order
        self.dt = dt
        self.step = step
        self.steps = steps
        # nu = hyperviscosity
        self.flux = fx.DGFlux(resolutions=resolutions, order=order, charge_mass=-1.0, nu=nu)
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

        # semi-implicit matrix
        self.inv_backward_advection = None
        self.build_advection_matrix(grid=grid)

        # save-times
        # self.save_times = np.array([10, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 130, 140, 150, 0, 0])
        # self.save_times = np.array([140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 160, 160, 160, 160, 0, 0])
        # self.save_times = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 0])
        # self.save_times = np.array([0, 10, 20, 30, 40, 50, 0])
        # self.save_times = np.array([72, 77, 82, 87, 92, 0])
        # self.save_times = np.append(np.linspace(100, 140, num=40), 0)
        # self.save_times = np.append(np.linspace(25, 75, num=250), 0)
        # self.save_times = np.append(np.linspace(140, 170, num=301), 0)
        # self.save_times = np.array([10, 20, 30, 40, 50, 0])
        # self.save_times = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 0])
        # self.save_times = np.array([60, 70, 80, 90, 100, 0])
        self.save_times = np.append(np.linspace(100, 150, num=250), 0)

    def main_loop_adams_bashforth(self, distribution, elliptic, grid, DataFile):  # , plotter, plot=True):
        """
        Evolve the Vlasov equation in wavenumber space using the Adams-Bashforth time integration scheme
        """
        print('Beginning main loop')
        # self.steps = 2

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

        # second stage
        self.ssp_rk3(distribution=distribution, elliptic=elliptic, grid=grid)

        # store first two fluxes
        previous_fluxes = [flux1, flux0]

        # Begin loop
        save_counter = 0
        for i in range(1, self.steps):
            previous_fluxes = self.adams_bashforth(distribution=distribution,
                                                   elliptic=elliptic, grid=grid, prev_fluxes=previous_fluxes)
            # experiment: enforce continuity on boundaries... a problem?
            # distribution.average_on_boundaries()
            self.time += self.step
            # print('Took a step')

            # Check out noise levels. Note: Incorrect to naively adapt time-step for adams-bashforth method
            if i % 50 == 0:
                self.time_array = np.append(self.time_array, self.time)
                elliptic.poisson_solve_single_species(distribution=distribution, grid=grid)
                self.field_energy = np.append(self.field_energy, elliptic.compute_field_energy(grid=grid))
                self.thermal_energy = np.append(self.thermal_energy,
                                                distribution.total_thermal_energy(grid=grid))
                self.density_array = np.append(self.density_array,
                                               distribution.total_density(grid=grid))
                # Max time-step velocity space
                elliptic.field.inverse_fourier_transform()
                max_field = cp.amax(elliptic.field.arr_nodal)
                max_dt = grid.v.min_dv / max_field / (2 * self.order + 1) / (2 * np.pi) * 0.01
                print('Took 50 steps, time is {:0.3e}'.format(self.time))
                # print('Max velocity-flux dt is {:0.3e}'.format(max_dt))

            if np.abs(self.time - self.save_times[save_counter]) < 6.0e-3:
                print('Reached save time at {:0.3e}'.format(self.time) + ', saving data...')
                DataFile.save_data(distribution=distribution.arr_nodal.get(),
                                   density=distribution.zero_moment.arr_nodal.get(),
                                   field=elliptic.field.arr_nodal.get(), time=self.time)
                save_counter += 1

    def ssp_rk3(self, distribution, elliptic, grid):
        # Cut-off (avoid CFL advection instability as this is fully explicit)
        cutoff = 500
        # Stage set-up
        stage0 = var.Distribution(resolutions=self.resolutions, order=self.order, charge_mass=None)
        stage1 = var.Distribution(resolutions=self.resolutions, order=self.order, charge_mass=None)

        # zero stage
        elliptic.poisson_solve_single_species(distribution=distribution, grid=grid)
        self.flux.semi_discrete_rhs(distribution=distribution, elliptic=elliptic, grid=grid)
        self.flux.output.arr[grid.x.device_modes > cutoff, :, :] = 0
        stage0.arr = distribution.arr + self.dt * self.flux.output.arr

        # first stage
        elliptic.poisson_solve_single_species(distribution=stage0, grid=grid)
        self.flux.semi_discrete_rhs(distribution=stage0, elliptic=elliptic, grid=grid)
        self.flux.output.arr[grid.x.device_modes > cutoff, :, :] = 0
        stage1.arr = (
                self.rk_coefficients[0, 0] * distribution.arr +
                self.rk_coefficients[0, 1] * stage0.arr +
                self.rk_coefficients[0, 2] * self.dt * self.flux.output.arr
        )

        # second stage
        elliptic.poisson_solve_single_species(distribution=stage1, grid=grid)
        self.flux.semi_discrete_rhs(distribution=stage1, elliptic=elliptic, grid=grid)
        self.flux.output.arr[grid.x.device_modes > cutoff, :, :] = 0
        distribution.arr = (
                self.rk_coefficients[1, 0] * distribution.arr +
                self.rk_coefficients[1, 1] * stage1.arr +
                self.rk_coefficients[1, 2] * self.dt * self.flux.output.arr
        )

    def adams_bashforth(self, distribution, elliptic, grid, prev_fluxes):
        # Compute Poisson constraint
        elliptic.poisson_solve_single_species(distribution=distribution, grid=grid)
        # Compute velocity flux
        self.flux.semi_discrete_rhs(distribution=distribution, elliptic=elliptic, grid=grid)
        # Update distribution according to explicit treatment of velocity flux and crank-nicholson for advection
        distribution.arr += self.dt * ((23 / 12 * self.flux.output.arr -
                                        4 / 3 * prev_fluxes[0] +
                                        5 / 12 * prev_fluxes[1]) +
                                       0.5 * self.flux.source_term_lgl_no_arr(distribution_arr=distribution.arr,
                                                                              grid=grid))
        # Do inverse half backward advection step
        distribution.arr = cp.einsum('nmjk,nmk->nmj', self.inv_backward_advection, distribution.arr)
        return [self.flux.output.arr, prev_fluxes[0]]

    def build_advection_matrix(self, grid):
        """ Construct the global backward advection matrix """
        backward_advection_operator = (cp.eye(grid.v.order)[None, None, :, :] -
                                       0.5 * self.dt * -1j * grid.x.device_wavenumbers[:, None, None, None] *
                                       grid.v.translation_matrix[None, :, :, :])
        self.inv_backward_advection = cp.linalg.inv(backward_advection_operator)


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
