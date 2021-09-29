import numpy as np
import scipy.integrate as spint
import time as timer
import fluxes as fx
import variables as var


nonlinear_ssp_rk_switch = {
    2: [[1 / 2, 1 / 2, 1 / 2]],
    3: [[3 / 4, 1 / 4, 1 / 4],
        [1 / 3, 2 / 3, 2 / 3]]
}


class Stepper:
    def __init__(self, dt, step, resolutions, order, steps):
        self.x_res, self.v_res = resolutions
        self.resolutions = resolutions
        self.order = order
        self.dt = dt
        self.step = step
        self.steps = steps
        self.flux = fx.DGFlux(resolutions=resolutions, order=order)

        # RK coefficients
        self.rk_coefficients = np.array(nonlinear_ssp_rk_switch.get(3, "nothing"))

        # tracking arrays
        self.time = 0
        self.next_time = 0
        self.field_energy = np.array([])
        self.time_array = np.array([])
        self.thermal_energy = np.array([])
        self.density_array = np.array([])

    def ode_system(self, t, y, distribution, elliptic, grid):
        distribution.arr = y.reshape(self.x_res, self.v_res, self.order)
        return self.flux.semi_discrete_rhs(distribution=distribution, elliptic=elliptic, grid=grid).flatten()

    def main_loop(self, distribution, elliptic, grid, plotter, plot=True):
        print('Beginning main loop')
        for i in range(self.steps):
            self.next_time = self.time + self.step
            span = [self.time, self.next_time]
            # sol = spint.solve_ivp(fun=self.ode_system, t_span=span, t_eval=[self.next_time],
            #                       y0=distribution.arr.flatten(), method='RK45', max_step=self.dt,
            #                       args=(distribution, elliptic, grid),
            #                       rtol=1.0e-12, atol=1.0e-12)
            # distribution.arr = sol.y.reshape(distribution.arr.shape)
            self.ssprk3(distribution=distribution, elliptic=elliptic, grid=grid)
            self.time += self.step
            self.time_array = np.append(self.time_array, self.time)
            elliptic.poisson_solve(distribution=distribution, grid=grid)
            self.field_energy = np.append(self.field_energy, elliptic.compute_field_energy(grid=grid))
            self.thermal_energy = np.append(self.thermal_energy, distribution.total_thermal_energy(grid=grid))
            self.density_array = np.append(self.density_array, distribution.total_density(grid=grid))
            print('Took step, time is {:0.3e}'.format(self.time))

            if plot:
                plotter.distribution_contourf(distribution=distribution, plot_spectrum=False)
                plotter.show()

        return distribution

    def ssprk3(self, distribution, elliptic, grid):
        stage0 = var.Distribution(resolutions=self.resolutions, order=self.order)
        stage1 = var.Distribution(resolutions=self.resolutions, order=self.order)
        # zero stage
        self.flux.semi_discrete_rhs(distribution=distribution, elliptic=elliptic, grid=grid)
        stage0.arr = distribution.arr + self.dt * self.flux.output.arr
        # first stage
        self.flux.semi_discrete_rhs(distribution=stage0, elliptic=elliptic, grid=grid)
        stage1.arr = (
                self.rk_coefficients[0, 0] * distribution.arr +
                self.rk_coefficients[0, 1] * stage0.arr +
                self.rk_coefficients[0, 2] * self.dt * self.flux.output.arr
        )
        # second stage
        self.flux.semi_discrete_rhs(distribution=stage1, elliptic=elliptic, grid=grid)
        distribution.arr = (
                self.rk_coefficients[1, 0] * distribution.arr +
                self.rk_coefficients[1, 1] * stage1.arr +
                self.rk_coefficients[1, 2] * self.dt * self.flux.output.arr
        )
