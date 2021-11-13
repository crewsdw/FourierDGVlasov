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
    return -0.5 * pd.Zdoubleprime(z_e) / fe / (np.sqrt(2) * k ** 3)


def eps_double_prime(k, z, mass_ratio, temperature_ratio, electron_drift):
    z_e = z - electron_drift / np.sqrt(2)
    fe = 1
    return -0.5 * pd.Ztripleprime(z_e) / fe / (2 * k ** 4)


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
mr = 1 / 1836  # / 10  # 1836  # me / mp
tr = 1.0  # Te / Tp
k = 0.5

e_d = 0  # 3.0

# grid
om_r = np.linspace(-0.2, 5, num=500)
om_i = np.linspace(-5, 0.1, num=500)

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
plt.xlabel('Real frequency')
plt.ylabel('Imaginary frequency')

plt.figure()
plt.contour(X, Y, np.real(eps), 0, colors='r')
plt.contour(X, Y, np.imag(eps), 0, colors='g')
plt.xlabel('Real frequency')
plt.ylabel('Imaginary frequency')
plt.tight_layout(), plt.grid(True), plt.show()

# guess_r, guess_i = (1.004, -0.01) / k_scale  # 1.43 / k_scale, -0.13 / k_scale

guesses_r = np.array([0, 1.4, 1.8, 2.2, 2.535, 2.83, 3.08, 3.31, 3.56, 3.80, 3.98]) / k_scale
guesses_i = np.array([0, -0.15, -1.12, -1.67, -2.09, -2.43, -2.72, -3.01, -3.24, -3.48, -3.70]) / k_scale

num = 10
sols = np.zeros_like(guesses_r) + 0j
guess = np.array([1.4156618886045 - 0.1533594669096j])
# guess = np.array([1.78957059-1.14414301j])
f0 = -2j * np.pi * (-guess * np.exp(-0.5 * guess ** 2.0)) / np.sqrt(2.0 * np.pi) / (k ** 2.0)

modes = np.arange(-11 + 1, 11)
prime = np.zeros_like(guesses_r) + 0j
amps = np.zeros_like(guesses_r) + 0j
conj_amps = np.zeros_like(guesses_r) + 0j

gauss_amps = np.zeros_like(guesses_r) + 0j
conj_gauss = np.zeros_like(guesses_r) + 0j

for i in range(11):
    solution = opt.root(dispersion_fsolve, x0=np.array([guesses_r[i], guesses_i[i]]), tol=1.0e-15,
                        args=(k, mr, tr, e_d), jac=jacobian_fsolve)
    sols[i] = (solution.x[0] + 1j * solution.x[1]) * k_scale
    print(sols[i])
    prime[i] = analytic_jacobian(k, sols[i] / k_scale, mr, tr, e_d)
    if i != 0:
        if i == 1:
            amps[i] = 1 - 0.5 * f0 * eps_double_prime(k, guess / k_scale, mr, tr, e_d) / (
                        analytic_jacobian(k, guess / k_scale, mr, tr, e_d) ** 2.0) / k ** 2
            conj = -1.0 * np.conj(sols[i])
            conj_amps[i] = f0 / ((guess - conj) * analytic_jacobian(k, conj / k_scale, mr, tr, e_d)) / k ** 2

            # for "gauss amps"
            gauss_amps[i] = pd.Z(sols[i] / k_scale) / analytic_jacobian(k, sols[i] / k_scale, mr, tr, e_d) / k ** 2
            conj_gauss[i] = pd.Z(conj / k_scale) / analytic_jacobian(k, conj / k_scale, mr, tr, e_d) / k ** 2
        if i != 1:
            amps[i] = f0 / ((guess - sols[i]) * analytic_jacobian(k, sols[i] / k_scale, mr, tr, e_d)) / k ** 2
            conj = -1.0 * np.conj(sols[i])
            conj_amps[i] = f0 / ((guess - conj) * analytic_jacobian(k, conj / k_scale, mr, tr, e_d)) / k ** 2

            # for gauss amps
            gauss_amps[i] = pd.Z(sols[i] / k_scale) / analytic_jacobian(k, sols[i] / k_scale, mr, tr, e_d) / k ** 2
            conj_gauss[i] = pd.Z(conj / k_scale) / analytic_jacobian(k, conj / k_scale, mr, tr, e_d) / k ** 2

amps[0] = 1
amps = np.append(np.append(np.flip(conj_amps[1:]), 0), amps[1:])
gamps = np.append(np.append(np.flip(conj_gauss[1:]), 0), gauss_amps[1:])

print(sols)

plt.figure()
plt.plot(np.real(sols), 'ro')
plt.plot(np.imag(sols), 'go')
plt.ylabel('Frequency')
plt.grid(True), plt.tight_layout()

plt.figure()
plt.plot(np.real(prime), 'ro')
plt.plot(np.imag(prime), 'go')
plt.ylabel('Derivative')
plt.grid(True), plt.tight_layout()

max = np.amax(np.absolute(amps))
gmax = np.amax(np.absolute(gamps))

plt.figure()
# plt.plot(modes, np.real(amps) / max, 'ro')
# plt.plot(modes, np.imag(amps) / max, 'go')
plt.plot(modes, np.absolute(amps) / max, 'ko')
plt.xlabel('Kinetic mode')
plt.ylabel('Modal amplitudes [arb. units]')
plt.title('Damping plane wave perturbation')
plt.axis([-10, 10, 0, 1.1])
plt.grid(True), plt.tight_layout()

plt.figure()
plt.plot(modes, np.absolute(gamps) / gmax, 'ko')
plt.xlabel('Kinetic mode'), plt.ylabel('Modal amplitudes [arb. units]')
plt.title('Gaussian perturbation')
plt.axis([-10, 10, 0, 1.1])
plt.grid(True), plt.tight_layout()

plt.show()

# zeta_sol = solution.x[0] + 1j * solution.x[1]
# print(analytic_jacobian(k, zeta_sol, mr, tr, e_d))
# print(dispersion_function(k, zeta_sol, mr, tr, e_d))

# x = (solution.x[0] + 1j * solution.x[1]) * k_scale
# print(f' x: {x:.16f}')
