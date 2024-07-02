//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "GPUIntegratedBC.h"

/**
 * Boundary condition for convective heat flux where temperature and heat transfer coefficient are
 * given by material properties.
 */
class GPUConvectiveHeatFluxBC final : public GPUIntegratedBC<GPUConvectiveHeatFluxBC>
{
public:
  static InputParameters validParams();

  GPUConvectiveHeatFluxBC(const InputParameters & parameters);

  KOKKOS_FUNCTION Real computeQpResidual(const unsigned int i,
                                         const unsigned int qp,
                                         const ResidualDatum & datum) const
  {
    return -datum.test(i, qp) * datum.getProperty(_htc, qp) *
           (datum.getProperty(_T_infinity, qp) - datum.u(qp));
  }
  KOKKOS_FUNCTION Real computeQpJacobian(const unsigned int i,
                                         const unsigned int j,
                                         const unsigned int qp,
                                         const ResidualDatum & datum) const
  {
    return -datum.test(i, qp) * datum.phi(j, qp) *
           (-datum.getProperty(_htc, qp) +
            datum.getProperty(_htc_dT, qp) * (datum.getProperty(_T_infinity, qp) - datum.u(qp)));
  }

private:
  /// Far-field temperature variable
  GPUMaterialProperty<Real> _T_infinity;

  /// Convective heat transfer coefficient
  GPUMaterialProperty<Real> _htc;

  /// Derivative of convective heat transfer coefficient with respect to temperature
  GPUMaterialProperty<Real> _htc_dT;
};
