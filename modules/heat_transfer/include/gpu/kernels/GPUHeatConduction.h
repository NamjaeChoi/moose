//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "GPUDiffusion.h"

class GPUHeatConduction final : public GPUDiffusion<GPUHeatConduction>
{
public:
  static InputParameters validParams();

  GPUHeatConduction(const InputParameters & parameters);

  KOKKOS_FUNCTION Real computeQpResidual(const unsigned int i,
                                         const unsigned int qp,
                                         const ResidualDatum & datum) const
  {
    return datum.getProperty(_diffusion_coefficient, qp) *
           GPUDiffusion::computeQpResidual(i, qp, datum);
  }
  KOKKOS_FUNCTION Real computeQpJacobian(const unsigned int i,
                                         const unsigned int j,
                                         const unsigned int qp,
                                         const ResidualDatum & datum) const
  {
    Real jac = datum.getProperty(_diffusion_coefficient, qp) *
               GPUDiffusion::computeQpJacobian(i, j, qp, datum);
    if (_diffusion_coefficient_dT)
      jac += datum.getProperty(_diffusion_coefficient_dT, qp) * datum.phi(j, qp) *
             GPUDiffusion::computeQpResidual(i, qp, datum);
    return jac;
  }

private:
  GPUMaterialProperty<Real> _diffusion_coefficient;
  GPUMaterialProperty<Real> _diffusion_coefficient_dT;
};
