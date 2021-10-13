import numpy as np
import scipy.integrate as spint
import time as timer
import fluxes as fx
import variables as var
import cupy as cp


nonlinear_ssp_rk_switch = {
    2: [[1 / 2, 1 / 2, 1 / 2]],
    3: [[3 / 4, 1 / 4, 1 / 4],
        [1 / 3, 2 / 3, 2 / 3]]
}


class Stepper:
    def __init__(self, dt, step, resolutions, order, steps, grids, charge_mass):
        self.x_res, self.v_res = resolutions
        self.resolutions = resolutions
        self.order = order
        self.dt = dt
        self.step = step
        self.steps = steps
        self.flux_e = fx.DGFlux(resolutions=resolutions, order=order, charge_mass=-1.0)
        self.flux_e.initialize_zero_pad(grid=grids[0])
        self.flux_p = fx.DGFlux(resolutions=resolutions, order=order, charge_mass=+1.0/charge_mass)
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

    def main_loop(self, distribution_e, distribution_p, elliptic, grids):  # , plotter, plot=True):
        print('Beginning main loop')
        for i in range(self.steps):
            self.next_time = self.time + self.step
            span = [self.time, self.next_time]
            self.ssp_rk3(distribution_e=distribution_e, distribution_p=distribution_p, elliptic=elliptic, grids=grids)
            self.time += self.step
            if i % 10 == 0:
                self.time_array = np.append(self.time_array, self.time)
                elliptic.poisson_solve(distribution_e=distribution_e, distribution_p=distribution_p, grids=grids)
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

    def ssp_rk3(self, distribution_e, distribution_p, elliptic, grids):
        # Stage set-up
        stage0_e = var.Distribution(resolutions=self.resolutions, order=self.order, charge_mass=None)
        stage0_p = var.Distribution(resolutions=self.resolutions, order=self.order, charge_mass=None)

        stage1_e = var.Distribution(resolutions=self.resolutions, order=self.order, charge_mass=None)
        stage1_p = var.Distribution(resolutions=self.resolutions, order=self.order, charge_mass=None)

        # zero stage
        elliptic.poisson_solve(distribution_e=distribution_e, distribution_p=distribution_p, grids=grids)
        self.flux_e.semi_discrete_rhs(distribution=distribution_e, elliptic=elliptic, grid=grids[0])
        self.flux_p.semi_discrete_rhs(distribution=distribution_p, elliptic=elliptic, grid=grids[1])

        stage0_e.arr = distribution_e.arr + self.dt * self.flux_e.output.arr
        stage0_p.arr = distribution_p.arr + self.dt * self.flux_p.output.arr

        # first stage
        elliptic.poisson_solve(distribution_e=stage0_e, distribution_p=stage0_p, grids=grids)
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
        elliptic.poisson_solve(distribution_e=stage1_e, distribution_p=stage1_p, grids=grids)
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

#
# def ode_system(self, t, y, distribution, elliptic, grid):
#     distribution.arr = y.reshape(self.x_res, self.v_res, self.order)
#     return self.flux.semi_discrete_rhs(distribution=distribution, elliptic=elliptic, grid=grid).flatten()

# sol = spint.solve_ivp(fun=self.ode_system, t_span=span, t_eval=[self.next_time],
#                       y0=distribution.arr.flatten(), method='RK45', max_step=self.dt,
#                       args=(distribution, elliptic, grid),
#                       rtol=1.0e-12, atol=1.0e-12)
# distribution.arr = sol.y.reshape(distribution.arr.shape)
