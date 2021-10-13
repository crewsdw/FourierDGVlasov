import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import scipy.optimize as opt
import plasma_dispersion as pd


def dispersion_function(k, z, mass_ratio, temperature_ratio):
    """
    Computes two-species plasma dispersion function epsilon(zeta, k) = 0
    """
    p_factor = mass_ratio * temperature_ratio
    k_sq = k ** 2.0
    z_e = z / np.sqrt(2)
    z_p = np.sqrt(temperature_ratio) / np.sqrt(2) * z
    return 1.0 - 0.5 * (pd.Zprime(z_e) + p_factor * pd.Zprime(z_p)) / k_sq


def analytic_jacobian(k, z, mass_ratio, temperature_ratio):
    p_factor = mass_ratio * temperature_ratio
    k_sq = k ** 2.0
    z_e = z / np.sqrt(2)
    z_p = np.sqrt(temperature_ratio) / np.sqrt(2) * z
    fe = 1 / np.sqrt(2)
    fp = np.sqrt(temperature_ratio) / np.sqrt(2)
    return -0.5 * (pd.Zdoubleprime(z_e) / fe + p_factor * pd.Zdoubleprime(z_p) / fp) / k_sq


def dispersion_fsolve(z, k, mass_ratio, temperature_ratio):
    freq = z[0] + 1j * z[1]
    d = dispersion_function(k, freq, mass_ratio, temperature_ratio)
    return [np.real(d), np.imag(d)]


def jacobian_fsolve(z, k, mass_ratio, temperature_ratio):
    freq = z[0] + 1j * z[1]
    jac = analytic_jacobian(k, freq, mass_ratio, temperature_ratio)
    jr, ji = np.real(jac), np.imag(jac)
    return [[jr, -ji], [ji, jr]]


# parameters
mr = 1/10  # / 10  # 1836
tr = 10.0
k = 0.5

# grid
om_r = np.linspace(-0.2, 2.2, num=500)
om_i = np.linspace(-1, 0.2, num=500)

k_scale = k * np.sqrt(2)

zr = om_r / k
zi = om_i / k

z = np.tensordot(zr, np.ones_like(zi), axes=0) + 1.0j * np.tensordot(np.ones_like(zr), zi, axes=0)

X, Y = np.meshgrid(om_r, om_i, indexing='ij')


eps = dispersion_function(k, z, mr, tr)
cb = np.linspace(-1, 1, num=100)

plt.figure()
plt.contourf(X, Y, np.real(eps), cb, extend='both')

plt.figure()
plt.contour(X, Y, np.real(eps), 0, colors='r')
plt.contour(X, Y, np.imag(eps), 0, colors='g')
plt.show()

guess_r, guess_i = 0.339 / k, -0.217 / k
solution = opt.root(dispersion_fsolve, x0=np.array([guess_r, guess_i]),
                    args=(k, mr, tr), jac=jacobian_fsolve)
print(solution.x * k_scale)
