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
    return 2 * pi * c / wavelength

def TBP_limited_length(wavelength, fwhm_wavelength):
    delta_nu = wavelength_fwhm_to_ang_freq_fwhm(wavelength, fwhm_wavelength) / (2 * pi)
    return 2 * np.log(2) / (pi * delta_nu)


def wavelength_fwhm_to_ang_freq_fwhm(wavelength, fwhm_wavelength):
    ang_freq_1 = wavelength_to_ang_freq(wavelength - fwhm_wavelength / 2)
    ang_freq_2 = wavelength_to_ang_freq(wavelength + fwhm_wavelength / 2)

    return np.abs(ang_freq_2 - ang_freq_1)

def gaussian_profile(t, a, b, c):
    return a * np.exp(-(t - b) ** 2 / (2 * c **2 ))

@dataclass
class pulse_stretch_class:
    center_lambda: float = 256e-9  # meter
    fwhm_lambda: float = 0.8e-9  # meter
    gdd_init: float = 0  # s^2

    n_points: int = 2 ** 15  # []
    t_max: float = 100e-12  # second

    def __post_init__(self):
        self.init_arrays()

    def __str__(self):
        str_str = f"Center Wavelength: {self.center_lambda*1e9} nm\n"
        str_str += f"Initial FWHM Wavelength: {self.fwhm_lambda*1e9} nm\n"
        str_str += f"Initial GDD: {self.gdd_init}\n"
        return str_str

    def init_arrays(self):
        t = np.linspace(-self.t_max, self.t_max, self.n_points)
        c = TBP_limited_length(self.center_lambda, self.fwhm_lambda)
        F = gaussian_profile(t, 1, 0, c)
        plt.plot(t, F)
        plt.show()
        



def main():
    """Main method."""
    test_class = pulse_stretch_class()

    print(test_class)


if __name__ == "__main__":
    main()

