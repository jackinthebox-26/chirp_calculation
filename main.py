from dataclasses import dataclass

import numpy as np

@dataclass
def class:
    center_lambda: float = 256e-9

    def __str__(self):
        str_str = f"Center Wavelength: {self.center_lambda*1e9} nm"
        return str_str




def main():
    """Main method."""
    test_class = class()

    print(test_class)


if __name__ == "__main__":
    main()

