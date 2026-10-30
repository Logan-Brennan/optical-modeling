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


def solve_bruggeman_spectrum(epsilon_components, volume_fractions):
    """
    Solve the Bruggeman effective-medium equation across a spectrum.

    Parameters
        epsilon_components are complex relative permittivities
        Volume fractions should sum to 1.

    Returns
        Effective complex relative permittivity at each wavelength.
    """

    epsilon_components = np.asarray(
        epsilon_components,
        dtype=complex
    )

    volume_fractions = np.asarray(
        volume_fractions,
        dtype=float
    )

    if epsilon_components.ndim != 2:
        raise ValueError(
            "epsilon_components must be a 2D array "
            "with shape (wavelengths, components)."
        )

    if epsilon_components.shape[1] != len(volume_fractions):
        raise ValueError(
            "Number of component spectra must match number of volume fractions."
        )

    num_wavelengths = epsilon_components.shape[0]

    epsilon_eff = np.empty(
        num_wavelengths,
        dtype=complex
    )

    for i in range(num_wavelengths):
        epsilon_eff[i] = solve_bruggeman(
            epsilon_components[i, :],
            volume_fractions
        )

    return epsilon_eff


def bruggeman_effective_nk(n_components, k_components, volume_fractions):
    """
    Calculate effective n and k spectra using the Bruggeman EMA.

    Parameters
    
    n_components - refractive-index spectra for each component material.

    k_components - extinction-coefficient spectra for each component material.

    volume_fractions - Volume fraction of each component material.
        Fractions should sum to 1.

    Returns
    
    n_eff : ndarray
        Effective refractive-index spectrum.

    k_eff : ndarray
        Effective extinction-coefficient spectrum.
    """

    n_components = np.column_stack(n_components)
    k_components = np.column_stack(k_components)

    if n_components.shape != k_components.shape:
        raise ValueError(
            "n_components and k_components must have matching shapes."
        )

    epsilon_components = nk_to_epsilon(
        n_components,
        k_components
    )

    epsilon_eff = solve_bruggeman_spectrum(
        epsilon_components,
        volume_fractions
    )

    n_eff, k_eff = epsilon_to_nk(epsilon_eff)

    return n_eff, k_eff