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
 * given by auxiliary variables.  Typically used in multi-app coupling scenario. It is possible to
 * couple in a vector variable where each entry corresponds to a "phase".
 */
class GPUCoupledConvectiveHeatFluxBC final : public GPUIntegratedBC<GPUCoupledConvectiveHeatFluxBC>
{
public:
  static InputParameters validParams();

  GPUCoupledConvectiveHeatFluxBC(const InputParameters & parameters);

  KOKKOS_FUNCTION Real computeQpResidual(const unsigned int i,
                                         const unsigned int qp,
                                         const ResidualDatum & datum) const
  {
    Real q = 0;
    Real u = datum.u(qp);
    for (std::size_t c = 0; c < _n_components; c++)
      q += datum.coupledValue(_alpha, qp, c) * datum.coupledValue(_htc, qp, c) *
           (u - datum.coupledValue(_T_infinity, qp, c));
    return datum.test(i, qp) * q * datum.coupledValue(_scale_factor, qp);
  }
  KOKKOS_FUNCTION Real computeQpJacobian(const unsigned int i,
                                         const unsigned int j,
                                         const unsigned int qp,
                                         const ResidualDatum & datum) const
  {
    Real dq = 0;
    Real phi = datum.phi(j, qp);
    for (std::size_t c = 0; c < _n_components; c++)
      dq += datum.coupledValue(_alpha, qp, c) * datum.coupledValue(_htc, qp, c) * phi;
    return datum.test(i, qp) * dq * datum.coupledValue(_scale_factor, qp);
  }

private:
  /// The number of components
  const unsigned int _n_components;
  /// Far-field temperature fields for each component
  GPUVariable _T_infinity;
  /// Convective heat transfer coefficient
  GPUVariable _htc;
  /// Volume fraction of individual phase
  GPUVariable _alpha;
  /// Scale factor
  GPUVariable _scale_factor;
};
