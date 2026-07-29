"""

This package uses a shifted carrier to make the calculations easier.


"""
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


pi = np.pi
c = sp.constants.c


def wavelength_to_ang_freq(wavelength):
    """
    Convert wavelength to angular frequency.

    Parameters
    ----------
    wavelength : float or array_like
        Wavelength in meters.

    Returns
    -------
    float or ndarray
        Angular frequency in radians per second.

    Raises
    ------
    ZeroDivisionError
        If `wavelength` is zero.

    Notes
    -----
    The relationship between wavelength and angular frequency is given by:

        omega = (2 * pi * c) / wavelength

    where `c` is the speed of light in a vacuum and `omega` is the angular
    frequency in rad/s.

    Examples
    --------
    >>> wavelength_to_ang_freq(500e-9)
    3.76610739343729e+15
    """
    return 2 * pi * c / wavelength

def TBP_limited_length(wavelength, fwhm_wavelength):
    """
    Calculate the transform-limited pulse duration for a Gaussian pulse.

    Parameters
    ----------
    wavelength : float or array_like
        Central wavelength in meters.
    fwhm_wavelength : float or array_like
        Full width at half maximum (FWHM) of the spectral wavelength in meters.

    Returns
    -------
    float or ndarray
        Transform-limited (TBP-limited) pulse duration (FWHM) in seconds.

    Notes
    -----
    For a Gaussian-shaped pulse, the lower bound of the Time-Bandwidth Product (TBP)
    relates the temporal duration (delta_t) and frequency bandwidth (delta_nu):

        delta_t * delta_nu = (2 * ln(2)) / pi ≈ 0.441

    This function rearranges the formula to solve for the shortest physically 
    achievable pulse duration (delta_t) given a spectral bandwidth.

    See Also
    --------
    wavelength_fwhm_to_ang_freq_fwhm : Convert spectral FWHM to angular frequency FWHM.
    """
    delta_nu = wavelength_fwhm_to_ang_freq_fwhm(wavelength, fwhm_wavelength) / (2 * pi)
    return 2 * np.log(2) / (pi * delta_nu)


def wavelength_fwhm_to_ang_freq_fwhm(wavelength, fwhm_wavelength):
    """
    Convert a wavelength spectral FWHM to an angular frequency FWHM.

    Calculates the spectral width in angular frequency by taking the absolute
    difference between the angular frequencies evaluated at the upper and 
    lower wavelength bounds.

    Parameters
    ----------
    wavelength : float or array_like
        Central wavelength in meters.
    fwhm_wavelength : float or array_like
        Full width at half maximum (FWHM) of the wavelength spectrum in meters.

    Returns
    -------
    float or ndarray
        Angular frequency FWHM in radians per second.

    See Also
    --------
    wavelength_to_ang_freq : Convert central wavelength to angular frequency.
    TBP_limited_length : Calculate transform-limited pulse duration using this FWHM.

    Notes
    -----
    This function calculates the exact angular frequency bandwidth from the upper
    and lower spectral limits rather than using the first-order derivative 
    approximation:

        delta_omega ≈ (2 * pi * c / wavelength**2) * delta_lambda
    """
    ang_freq_1 = wavelength_to_ang_freq(wavelength - fwhm_wavelength / 2)
    ang_freq_2 = wavelength_to_ang_freq(wavelength + fwhm_wavelength / 2)

    return np.abs(ang_freq_2 - ang_freq_1)

def gaussian_profile(t, a, b, c):
    """
    Calculate a Gaussian profile.

    Parameters
    ----------
    t : float or array_like
        Independent variable (e.g., time or position).
    a : float
        Amplitude (peak height) of the Gaussian.
    b : float
        Center position (mean or peak location) of the Gaussian.
    c : float
        Width parameter (standard deviation, sigma) of the Gaussian.

    Returns
    -------
    float or ndarray
        Evaluated Gaussian profile at `t`.

    Notes
    -----
    The Gaussian profile is defined as:

        f(t) = a * exp(-(t - b)**2 / (2 * c**2))

    The Full Width at Half Maximum (FWHM) is related to parameter `c` by:

        FWHM = 2 * sqrt(2 * ln(2)) * c ≈ 2.3548 * c
    """
    return a * np.exp(-(t - b) ** 2 / (2 * c **2 ))

def time_to_freq_fft(t, f):
    """
    Convert a time-domain field envelope to the frequency domain using the FFT.

    Parameters
    ----------
    t : array_like
        Equally spaced 1D array of time points in seconds.
    f : array_like
        1D real or complex array representing the electric field envelope in 
        the time domain.

    Returns
    -------
    omega : ndarray
        1D array of relative angular frequencies in radians per second (rad/s), 
        centered at zero.
    E_omega : ndarray
        1D complex array representing the spectral envelope in the frequency domain.

    Notes
    -----
    This function approximates the continuous Fourier transform:

        E(omega) = integral_{-inf}^{inf} f(t) * exp(-i * omega * t) dt

    Key numerical details applied:
    - `np.fft.ifftshift(f)` shifts t = 0 to index 0 prior to FFT to eliminate 
      artificial linear phase slopes.
    - `np.fft.fftshift` centers zero angular frequency (omega = 0) at the array midpoint.
    - Multiplication by time step `dt` scales the discrete transform to preserve 
      physical units.
    """
    n_points = len(t)
    dt = t[1] - t[0]

    freqs = np.fft.fftfreq(n_points, d=dt)
    omega = 2 * pi * np.fft.fftshift(freqs)

    E_omega = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(f))) * dt

    return omega, E_omega

def freq_to_time_fft(omega, F):
    """
    Convert a spectral envelope from frequency domain to time domain using IFFT.

    Parameters
    ----------
    omega : array_like
        Equally spaced 1D array of relative angular frequencies in rad/s,
        centered at zero.
    F : array_like
        1D complex array representing the spectral envelope in the frequency domain.

    Returns
    -------
    t : ndarray
        1D array of time points in seconds, centered at zero.
    f : ndarray
        1D complex array representing the field envelope in the time domain.

    Notes
    -----
    This function evaluates the continuous inverse Fourier transform:

        f(t) = (1 / (2 * pi)) * integral_{-inf}^{inf} F(omega) * exp(i * omega * t) d_omega

    Detailed breakdown of the transformation chain 
    `np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(F))) / dt`:

    1. `np.fft.ifftshift(F)`
       Un-shifts the input spectrum `F`. Standard FFT functions expect the 0 Hz bin 
       at index 0, but `F` is centered (0 Hz at index N//2). `ifftshift` moves 
       zero frequency back to index 0 so the IFFT algorithm interprets phase 
       and frequency correctly.

    2. `np.fft.ifft(...)`
       Executes the Inverse Discrete Fourier Transform (IDFT) on the un-shifted 
       spectrum, converting the frequency data back into the time domain.

    3. `np.fft.fftshift(...)`
       Shifts the resulting time-domain output array so that t = 0 is moved 
       from index 0 back to the center of the array (index N//2), aligning with 
       the centered time vector `t`.

    4. `/ dt`
       Normalizes the discrete output. Dividing by `dt` inverts the `* dt` scaling 
       applied in `time_to_freq_fft`, matching continuous Fourier integral units 
       and ensuring exact amplitude/energy conservation during round trips.
    """
    n_points = len(omega)
    d_omega = omega[1] - omega[0]

    df = d_omega / (2 * pi)
    dt = 1 / (n_points * df)
    t = np.fft.fftshift(np.fft.fftfreq(n_points, d=df))

    f = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(F))) / dt

    return t, f

@dataclass
class pulse_stretch_class:
    center_lambda: float = 256e-9  # meter
    fwhm_lambda: float = 0.8e-9  # meter
    gdd_init: float = 0  # s^2

    n_points: int = 2 ** 15  # []
    t_max: float = 100e-12  # second

    def __post_init__(self):
        self.t, self.f, self.omega, self.F = self.init_arrays()

    def __str__(self):
        str_str = f"Center Wavelength: {self.center_lambda*1e9} nm\n"
        str_str += f"Initial FWHM Wavelength: {self.fwhm_lambda*1e9} nm\n"
        str_str += f"Initial GDD: {self.gdd_init}\n"
        return str_str

    def init_arrays(self):
        # Time Domain
        t = np.linspace(-self.t_max, self.t_max, self.n_points)
        dt = t[1] - t[0]
        c = TBP_limited_length(self.center_lambda, self.fwhm_lambda) / (2 * np.sqrt(np.log(2)))
        f = gaussian_profile(t, 1, 0, c).astype(np.complex128)

        # Frequency Domain
        omega, F = time_to_freq_fft(t, f)

        return t, f, omega, F 

    def apply_GDD(self, GDD):
        F_in = self.F
        F_out = F_in * np.exp(-1j * GDD * self.omega ** 2 / 2)

        self.F = F_out

        self.t, self.f = freq_to_time_fft(self.omega, self.F)



def main():
    """Main method."""
    test_class = pulse_stretch_class()

    print(test_class)


if __name__ == "__main__":
    main()

