import numpy as np
import cupy as cp
import scipy.optimize as opt
import matplotlib.pyplot as plt
import time as timer


def solve_approximate_dielectric_function(distribution, grid_v, grid_k):
    # Compute p.v. integral via Hilbert transform of distribution
    distribution.fourier_transform(grid=grid_v)
    distribution.fourier_grad(grid=grid_v)
    pv_integral = distribution.hilbert_transform_grad(grid=grid_v)

    # initialize arrays
    # grid_k = np.linspace(0.17, 0.4, num=100)
    solutions = np.zeros_like(grid_k)
    growth_rates = np.zeros_like(solutions)
    # approx_plasma = np.zeros_like(solutions)
    # om_bohmgross = np.zeros_like(solutions)
    # approx_bohmgross = np.zeros_like(solutions)
    guess = 5.84  # 6.1  # 5.6
    # Check out for various grid_k frequencies
    for idx, wave in enumerate(grid_k):
        # dielectric = wave ** 2.0 - pv_integral  # / (wave ** 2.0)
        dielectric = 1 - pv_integral / (wave ** 2.0)
        if cp.amin(dielectric) > 0:
            continue

        print('Examining wave {:0.2f}'.format(wave))
        print('Guess is ' + str(guess))

        # plt.figure()
        # # plt.plot(grid_v.arr.flatten(), pv_integral.flatten().get(), 'o--')
        # plt.plot(grid_v.arr.flatten(), dielectric.get().flatten(), 'o--')
        # plt.plot([guess, guess], [-5, 5], '--')
        # plt.grid(True), plt.tight_layout(), plt.show()

        def interpolated_dielectric(phase_velocity):
            # print(phase_velocity)
            phase_velocity = phase_velocity[0]
            vidx, velocity = grid_v.get_local_velocity(phase_velocity=
                                                       phase_velocity)
            interpolant_on_point = grid_v.get_interpolant_on_point(velocity=velocity)
            dielectric_on_point = np.tensordot(dielectric[vidx, :].get(), interpolant_on_point, axes=([1], [0]))
            return dielectric_on_point[0]

        def interpolated_dielectric_grad(phase_velocity):
            vidx, velocity = grid_v.get_local_velocity(phase_velocity=
                                                       phase_velocity)
            interpolant_grad_on_point = grid_v.get_interpolant_grad_on_point(velocity=velocity)
            dielectric_grad_on_point = np.tensordot(dielectric[vidx, :].get(), interpolant_grad_on_point,
                                                    axes=([1], [0]))
            return dielectric_grad_on_point[0]

        def grad_on_point(phase_velocity):
            vidx, velocity = grid_v.get_local_velocity(phase_velocity=
                                                       phase_velocity)
            interpolant_grad_on_point = grid_v.get_interpolant_grad_on_point(velocity=velocity)
            return np.tensordot(distribution.arr[vidx, :], interpolant_grad_on_point, axes=([1], [0]))[0]

        # solve it
        solutions[idx] = opt.fsolve(func=interpolated_dielectric, x0=np.array(guess))  # ,
                                    # fprime=interpolated_dielectric_grad)
        growth_rates[idx] = np.pi * (grad_on_point(phase_velocity=solutions[idx]) /
                                     interpolated_dielectric_grad(phase_velocity=solutions[idx])) / wave ** 2.0
        guess = solutions[idx]

    # reshape solutions
    solutions = solutions.reshape(grid_k.shape)
    growth_rates = growth_rates.reshape(grid_k.shape)
    # growth_rates[(growth_rates < 0) & (np.abs(growth_rates) > 2 * np.amax(growth_rates))] = -2 * np.amax(growth_rates)

    plt.figure()
    plt.plot(grid_v.arr.flatten(), distribution.arr.flatten(), linewidth=3)
    plt.xlabel(r'Velocity $v/v_t$'), plt.ylabel(r'Average distribution $\langle f\rangle_L$')
    plt.grid(True), plt.tight_layout()


    # plt.figure()
    # plt.plot(grid_k, solutions, 'ro--')
    # plt.plot(grid_k, growth_rates, 'go--')
    # plt.grid(True), plt.tight_layout()
    #
    # plt.figure()
    # plt.plot(grid_k, solutions * grid_k, 'ro--', label=r'Real frequency')
    # plt.plot(grid_k, growth_rates * grid_k, 'go--', label=r'Growth rate')
    # plt.xlabel(r'Wavenumber $k\lambda_D$'), plt.ylabel(r'Frequency $\omega/\omega_p$')
    # plt.grid(True), plt.legend(loc='best'), plt.tight_layout()

    # Estimate group velocities
    om = solutions * grid_k
    dk = grid_k[1] - grid_k[0]
    group_vel = np.zeros_like(solutions)
    group_vel[1:-1] = (om[2:] - om[:-2]) / (2 * dk)
    group_vel[0] = (om[1] - om[0]) / dk
    group_vel[-1] = (om[-1] - om[-2]) / dk

    # plt.figure()
    # plt.plot(grid_k, group_vel, 'o--')
    # plt.xlabel(r'Wavenumber $k\lambda_D$'), plt.ylabel(r'Group velocity $v_g/v_t$')
    # plt.grid(True), plt.legend(loc='best'), plt.tight_layout()

    ac_freq = np.abs(0.01 * (group_vel - solutions))  # autocorrelation frequency (inverse wavepacket lifetime)

    # plt.figure()
    # plt.plot(grid_k, 1/ac_freq, 'o--')
    # plt.xlabel(r'Wavenumber $k\lambda_D$'), plt.ylabel(r'Wavepacket autocorrelation '
    #                                                    r'$\tau_{ac} = |\Delta k (v_g - v_\varphi)|^{-1}$')
    # plt.grid(True), plt.legend(loc='best'), plt.tight_layout()

    return solutions, growth_rates