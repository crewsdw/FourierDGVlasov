import numpy as np
import scipy.integrate as spint
import time as timer
import fluxes as fx


class Stepper:
    def __init__(self, dt, step, resolutions, order, steps):
        self.x_res, self.v_res = resolutions
        self.order = order
        self.dt = dt
        self.step = step
        self.steps = steps
        self.flux = fx.DGFlux(resolutions=resolutions, order=order)
        self.time = 0
        self.next_time = 0
        self.field_energy = np.array([])
        self.time_array = np.array([])

    def ode_system(self, t, y, distribution, elliptic, grid):
        distribution.arr = y.reshape(self.x_res, self.v_res, self.order)
        # print('I am evaling, and t is {:0.3e}'.format(t))
        return self.flux.semi_discrete_rhs(distribution=distribution, elliptic=elliptic, grid=grid).flatten()

    def main_loop(self, distribution, elliptic, grid, plotter, plot=True):
        print('Beginning main loop')
        for i in range(self.steps):
            self.next_time = self.time + self.step
            span = [self.time, self.next_time]
            sol = spint.solve_ivp(fun=self.ode_system, t_span=span, t_eval=[self.next_time],
                                  y0=distribution.arr.flatten(), method='RK45', max_step=self.dt,
                                  args=(distribution, elliptic, grid),
                                  rtol=1.0e-12, atol=1.0e-12)
            distribution.arr = sol.y.reshape(distribution.arr.shape)
            self.time += self.step
            self.time_array = np.append(self.time_array, self.time)
            self.field_energy = np.append(self.field_energy, elliptic.compute_field_energy(grid=grid))
            print('Took step, time is {:0.3e}'.format(self.time))

            if plot:
                plotter.distribution_contourf(distribution=distribution, plot_spectrum=False)
                plotter.show()

        return distribution
