from dataclasses import dataclass

import numpy as np

@dataclass
class pulse_stretch_class:
    center_lambda: float = 256e-9
    fwhm_lambda: float = 0.8e-9
    gdd_init: float = 0

    def __str__(self):
        str_str = f"Center Wavelength: {self.center_lambda*1e9} nm\n"
        str_str += f"FWHM Wavelength: {self.fwhm_lambda*1e9} nm\n"
        str_str += f"Initial GDD: {self.gdd_init}\n"
        return str_str




def main():
    """Main method."""
    test_class = pulse_stretch_class()

    print(test_class)


if __name__ == "__main__":
    main()

