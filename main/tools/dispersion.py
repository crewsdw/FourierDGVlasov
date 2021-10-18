import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import scipy.optimize as opt
import tools.plasma_dispersion as pd


def dispersion_function(k, z, mass_ratio, temperature_ratio, electron_drift):
    """
    Computes two-species plasma dispersion function epsilon(zeta, k) = 0
    """
    thermal_velocity_ratio = np.sqrt(temperature_ratio / mass_ratio)
    k_sq = k ** 2.0
    z_e = z - electron_drift / np.sqrt(2)
    z_p = thermal_velocity_ratio * z
    # return 1.0 - 0.5 * (pd.Zprime(z_e) + temperature_ratio * pd.Zprime(z_p)) / k_sq
    return 1.0 - pd.Zprime(z_e) / k_sq / 2


def analytic_jacobian(k, z, mass_ratio, temperature_ratio, electron_drift):
    thermal_velocity_ratio = np.sqrt(temperature_ratio / mass_ratio)
    # k_sq = k ** 2.0
    z_e = z - electron_drift / np.sqrt(2)
    z_p = thermal_velocity_ratio * z
    fe = 1
    fp = thermal_velocity_ratio
    # return -0.5 * (pd.Zdoubleprime(z_e) / fe + temperature_ratio * pd.Zdoubleprime(z_p) / fp) / k_sq
    return -0.5 * pd.Zdoubleprime(z_e) / fe / (k ** 3)


def dispersion_fsolve(z, k, mass_ratio, temperature_ratio, electron_drift):
    freq = z[0] + 1j * z[1]
    d = dispersion_function(k, freq, mass_ratio, temperature_ratio, electron_drift)
    return [np.real(d), np.imag(d)]


def jacobian_fsolve(z, k, mass_ratio, temperature_ratio, electron_drift):
    freq = z[0] + 1j * z[1]
    jac = analytic_jacobian(k, freq, mass_ratio, temperature_ratio, electron_drift)
    jr, ji = np.real(jac), np.imag(jac)
    return [[jr, -ji], [ji, jr]]


# parameters
mr = 1/1836  # / 10  # 1836  # me / mp
tr = 1.0  # Te / Tp
k = 1

e_d = 0  # 3.0

# grid
om_r = np.linspace(-3.2, 5.2, num=500)
om_i = np.linspace(-4, 1.0, num=500)

k_scale = k * np.sqrt(2)

zr = om_r / k_scale
zi = om_i / k_scale

z = np.tensordot(zr, np.ones_like(zi), axes=0) + 1.0j * np.tensordot(np.ones_like(zr), zi, axes=0)

X, Y = np.meshgrid(om_r, om_i, indexing='ij')


eps = dispersion_function(k, z, mr, tr, e_d)
eps_prime = analytic_jacobian(k, z, mr, tr, e_d)

cb = np.linspace(-1, 1, num=100)

plt.figure()
plt.contourf(X, Y, np.real(eps), cb, extend='both')

plt.figure()
plt.contourf(X, Y, np.real(eps_prime), cb, extend='both')

plt.figure()
plt.contour(X, Y, np.real(eps), 0, colors='r')
plt.contour(X, Y, np.imag(eps), 0, colors='g')
plt.show()

guess_r, guess_i = (2.0, -0.8) / k_scale  # 1.43 / k_scale, -0.13 / k_scale
solution = opt.root(dispersion_fsolve, x0=np.array([guess_r, guess_i]), tol=1.0e-15,
                    args=(k, mr, tr, e_d), jac=jacobian_fsolve)

zeta_sol = solution.x[0] + 1j * solution.x[1]
print(analytic_jacobian(k, zeta_sol, mr, tr, e_d))
print(dispersion_function(k, zeta_sol, mr, tr, e_d))

x = (solution.x[0] + 1j * solution.x[1]) * k_scale
print(f' x: {x:.16f}')
