import numpy as np

def energy_to_wavelength(energy_ev):
    """
    Convert photon energy in eV to wavelength in nm
    """
    energy_ev = np.asarray(energy_ev, dtype=float)

    if np.any(energy_ev <= 0):
        raise ValueError("Photon energies must be greater than zero")

    wavelength_nm = 1239.841984 / energy_ev

    return wavelength_nm


def sort_spectrum(wavelength, n, k):
    """
    Sort wavelength-dependent optical constants by increasing wavelength
    """
    wavelength_nm = np.asarray(wavelength, dtype=float)
    n = np.asarray(n, dtype=float)
    k = np.asarray(k, dtype=float)

    if len(wavelength_nm) != len(n) or len(wavelength_nm) != len(k):
        raise ValueError("wavelength, n, and k must have the same length")
    if np.any(wavelength_nm <=0 ):
        raise ValueError("Wavelengths must be greater than zero")
    if not (
        np.all(np.isfinite(wavelength_nm))
        and np.all(np.isfinite(n))
        and np.all(np.isfinite(k))
    ):
        raise ValueError("wavelength, n, and k must contain only finite values")

    sort_indices = np.argsort(wavelength_nm)

    wavelength_nm = wavelength_nm[sort_indices]
    n = n[sort_indices]
    k = k[sort_indices]

    return wavelength_nm, n, k