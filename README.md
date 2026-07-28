# README

This class is to calculate the stretching in a UV pulse as it passes through objects. 

When the class is initiated, we assume that the pulse is transform limited. 
Then using fourier transforms, a new pulse is generated using the specified GDD.
Then the pulse length can be calculated.
If the pulse information is saved, then another instance of GDD can be applied. 


## Inputs
- Center wavelength
- Wavelength FWHM

## Outputs
- Pulse length
- New pulse in the time domain.
