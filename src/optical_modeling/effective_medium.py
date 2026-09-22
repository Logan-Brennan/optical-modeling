import numpy as np

def nk_to_epsilon(n, k):
    """
    Convert refractive index (n, k) to complex dielectric constant.
    epsilon = (n + ik)^2, using n_complex = n + ik.
    """
    n_complex = n + 1j * k
    epsilon = n_complex ** 2
    return epsilon

def epsilon_to_nk(epsilon):
    """Convert complex dielectric constant to refractive index (n, k)."""
    epsilon_real = epsilon.real
    epsilon_imag = epsilon.imag
    epsilon_magnitude = np.sqrt(epsilon_real**2 + epsilon_imag**2)
    n = np.sqrt((epsilon_magnitude + epsilon_real) / 2.0)
    k = np.sqrt((epsilon_magnitude - epsilon_real) / 2.0)
    return n, k

