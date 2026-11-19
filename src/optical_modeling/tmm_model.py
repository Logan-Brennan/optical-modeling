import numpy as np
from tmm import inc_tmm


def calculate_rt_spectrum(
    wavelength_nm,
    n_spectra,
    d_list,
    c_list,
    angle_deg=0.0,
    polarization="unpolarized",
):
    """
    Calculate reflectance and transmittance spectra for a multilayer stack.

    Parameters
    ----------
    wavelength_nm : array-like
        Wavelength grid in nm.

    n_spectra : list of arrays
        Complex refractive-index spectrum for each layer.
        Layers must be listed in the order encountered by the light.

    d_list : list
        Layer thicknesses in nm.
        First and last layers must have infinite thickness.

    c_list : list
        Coherency of each layer:
        'c' = coherent
        'i' = incoherent

    angle_deg : float
        Angle of incidence in degrees in the incident medium.

    polarization : str
        'unpolarized', 's', or 'p'.

    Returns
    -------
    R : ndarray
        Reflectance spectrum.

    T : ndarray
        Transmittance spectrum.
    """

    wavelength_nm = np.asarray(
        wavelength_nm,
        dtype=float
    )

    n_spectra = [
        np.asarray(n, dtype=complex)
        for n in n_spectra
    ]

    if not (
        len(n_spectra)
        == len(d_list)
        == len(c_list)
    ):
        raise ValueError(
            "n_spectra, d_list, and c_list must have the same length."
        )

    for n in n_spectra:
        if len(n) != len(wavelength_nm):
            raise ValueError(
                "Each refractive-index spectrum must match "
                "the wavelength grid length."
            )

    if polarization not in [
        "unpolarized",
        "s",
        "p",
    ]:
        raise ValueError(
            "polarization must be 'unpolarized', 's', or 'p'."
        )

    angle_rad = np.deg2rad(angle_deg)

    R = np.empty(
        len(wavelength_nm),
        dtype=float
    )

    T = np.empty(
        len(wavelength_nm),
        dtype=float
    )

    for i, wavelength in enumerate(wavelength_nm):

        n_list = [
            n[i]
            for n in n_spectra
        ]

        if polarization == "unpolarized" and np.isclose(angle_deg, 0.0):

            result = inc_tmm(
                "s",
                n_list,
                d_list,
                c_list,
                angle_rad,
                wavelength,
            )

            R[i] = result["R"]
            T[i] = result["T"]

        elif polarization == "unpolarized":

            result_s = inc_tmm(
                "s",
                n_list,
                d_list,
                c_list,
                angle_rad,
                wavelength,
            )

            result_p = inc_tmm(
                "p",
                n_list,
                d_list,
                c_list,
                angle_rad,
                wavelength,
            )

            R[i] = 0.5 * (
                result_s["R"]
                + result_p["R"]
            )

            T[i] = 0.5 * (
                result_s["T"]
                + result_p["T"]
            )

        else:

            result = inc_tmm(
                polarization,
                n_list,
                d_list,
                c_list,
                angle_rad,
                wavelength,
            )

            R[i] = result["R"]
            T[i] = result["T"]

    return R, T