# JouleHeatingHeatGeneratedAux

!syntax description /AuxKernels/JouleHeatingHeatGeneratedAux

## Description

The `JouleHeatingHeatGeneratedAux` AuxKernel is used to compute the heat generated by Joule heating. The heat (power per unit volume) is computed as $dP/dV= \bm{J} \cdot \bm{E} = E ^2 \sigma$, where $\bm{J} = \sigma \bm{E}$ is the current density, $\bm{E}$ is the electric field, and $ \sigma $ is the electrical conductivity.

!syntax parameters /AuxKernels/JouleHeatingHeatGeneratedAux

!syntax inputs /AuxKernels/JouleHeatingHeatGeneratedAux

!syntax children /AuxKernels/JouleHeatingHeatGeneratedAux
