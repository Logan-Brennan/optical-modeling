import numpy as np
from scipy.optimize import fsolve

def nk_to_epsilon(n, k):
    """
    Convert refractive index (n, k) to complex relative permittivity.
    epsilon = (n + ik)^2, using n_complex = n + ik.
    """
    n_complex = n + 1j * k
    epsilon = n_complex ** 2
    return epsilon

def epsilon_to_nk(epsilon):
    """Convert complex relative permittivity to refractive index (n, k)."""
    epsilon_real = epsilon.real
    epsilon_imag = epsilon.imag
    epsilon_magnitude = np.sqrt(epsilon_real**2 + epsilon_imag**2)
    n = np.sqrt((epsilon_magnitude + epsilon_real) / 2.0)
    k = np.sqrt((epsilon_magnitude - epsilon_real) / 2.0)
    return n, k

def bruggeman_residual(epsilon_eff_array, epsilon_components, volume_fractions):

    epsilon_eff = epsilon_eff_array[0] + 1j * epsilon_eff_array[1]

    epsilon_components = np.asarray(epsilon_components, dtype=complex)
    volume_fractions = np.asarray(volume_fractions, dtype=float)
    terms = volume_fractions * (
        (epsilon_components - epsilon_eff) 
        / (epsilon_components + 2 * epsilon_eff)
    )

    total = np.sum(terms)
    return [total.real, total.imag]

def solve_bruggeman(epsilon_components, volume_fractions):
    """
    Solve the Bruggeman effective-medium equation at a single wavelength.
    
    Parameters: 

    epsilon_components : array-like
    Complex relative permittivities of the component materials at a single wavelength.

    volume_fractions : array-like
    Volume fractions corresponding to each component material. Fractions should sum to 1.
    """
    epsilon_components = np.asarray(epsilon_components, dtype=complex)
    volume_fractions = np.asarray(volume_fractions, dtype=float)

    if len(epsilon_components) != len(volume_fractions):
        raise ValueError("epsilon_components and volume_fractions must have the same length.")

    if np.any(volume_fractions < 0):
        raise ValueError("volume fractions cannot be negative.")

    if not np.isclose(np.sum(volume_fractions), 1.0):
        raise ValueError("volume fractions must sum to 1.")

    epsilon_guess = np.sum(volume_fractions * epsilon_components)
    initial_guess = [epsilon_guess.real, epsilon_guess.imag]

    solution, info, ier, message = fsolve(bruggeman_residual, initial_guess, args=(epsilon_components, volume_fractions), full_output=True)
    if ier != 1:
        raise RuntimeError(f"Bruggeman solver failed: {message}")

    epsilon_eff = solution[0] + 1j * solution[1]

    return epsilon_eff