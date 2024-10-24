# GradField

!syntax description /Kernels/GradField

## Overview

!style halign=left
The GradField object implements the following PDE term for coupled
scalar-vector PDE systems:

\begin{equation}
  - k \nabla u,
\end{equation}

where $k$ is a constant scalar coefficient and  $u$ is a scalar field variable.
Given vector test functions $\vec{\psi_i}$, the weak form, in inner-product
notation, is given by:

\begin{equation}
  R_i(u) = (\nabla \cdot \vec{\psi_i}, k u) \quad \forall \vec{\psi_i}.
\end{equation}

## Example Input File Syntax

!listing coupled_electrostatics.i block=Kernels/gradient

!syntax parameters /Kernels/GradField

!syntax inputs /Kernels/GradField

!syntax children /Kernels/GradField
