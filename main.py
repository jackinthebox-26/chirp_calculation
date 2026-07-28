from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

@dataclass
class pulse_stretch_class:
    center_lambda: float = 256e-9  # meter
    fwhm_lambda: float = 0.8e-9  # meter
    gdd_init: float = 0  # s^2

    n_points: int = 2 ** 5  # []
    t_max: float = 1e-9  # second

    def __post_init__(self):
        self.init_arrays()

    def __str__(self):
        str_str = f"Center Wavelength: {self.center_lambda*1e9} nm\n"
        str_str += f"FWHM Wavelength: {self.fwhm_lambda*1e9} nm\n"
        str_str += f"Initial GDD: {self.gdd_init}\n"
        return str_str

    def init_arrays(self):
        t = np.linspace(-self.t_max, self.t_max, self.n_points)
        F = t
        plt.plot(t, F)
        plt.show()
        



def main():
    """Main method."""
    test_class = pulse_stretch_class()

    print(test_class)


if __name__ == "__main__":
    main()

