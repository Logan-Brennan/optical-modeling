import numpy as np


#------------
# User inputs
# -----------

start_nm = 400
end_nm = 1000
step_nm = 1

#-------------------
# End of user inputs
#-------------------

def prepare_optical_data(data, wavelength_grid):
    """
    Standardize optical-constant data with columns:
    'en' (energy in eV), 'n', and 'k'
    """
    energy_ev = data["en"].to_numpy()
    n = data["n"].to_numpy()
    k = data["k"].to_numpy()

    wavelength_nm = energy_to_wavelength(energy_ev)

    wavelength_nm, n, k = sort_spectrum(
        wavelength_nm,
        n,
        k,
    )

    n_interp, k_interp = interpolate_spectrum(
        wavelength_nm,
        n,
        k,
        wavelength_grid,
    )

    return n_interp, k_interp

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

    if np.any(np.diff(wavelength_nm) == 0):
        raise ValueError("duplicate wavelength values found")

    return wavelength_nm, n, k

def make_wavelength_grid(start_nm, end_nm, step_nm):
    """Create a wavelength grid over a user-specified range"""

    start_nm = float(start_nm)
    end_nm = float(end_nm)
    step_nm = float(step_nm)

    if start_nm <= 0 or end_nm <0:
        raise ValueError("Wavelengths must be greater than zero")

    if end_nm <= start_nm:
        raise ValueError("end_nm must be greater than start_nm")
    
    if step_nm <= 0:
        raise ValueError("step_nm must be greater than zero")

    wavelength_grid = np.arange(start_nm, end_nm + step_nm, step_nm)

    return wavelength_grid

def interpolate_spectrum(wavelength_nm, n, k, wavelength_grid):
    """Interpolate n and k onto a specified wavelength grid"""

    wavelength_nm, n ,k = sort_spectrum(wavelength_nm, n, k)

    wavelength_grid = np.asarray(wavelength_grid, dtype=float)

    if (np.min(wavelength_grid) < np.min(wavelength_nm)
        or np.max(wavelength_grid) > np.max(wavelength_nm)
        ):
        raise ValueError("Requested wavelength grid extends beyond available spectrum")

    n_interp = np.interp(wavelength_grid, wavelength_nm, n)
    k_interp = np.interp(wavelength_grid, wavelength_nm, k)

    return n_interp, k_interp