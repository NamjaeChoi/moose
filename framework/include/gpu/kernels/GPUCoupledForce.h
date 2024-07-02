//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "GPUKernel.h"

/**
 * Implements a source term proportional to the value of a coupled variable. Weak form: $(\\psi_i,
 * -\\sigma v)$.
 */
class GPUCoupledForce : public GPUKernel<GPUCoupledForce>
{
public:
  static InputParameters validParams();

  GPUCoupledForce(const InputParameters & parameters);

  KOKKOS_FUNCTION Real computeQpResidual(const unsigned int i,
                                         const unsigned int qp,
                                         const ResidualDatum & datum) const
  {
    return -_coef * datum.coupledValue(_v, qp) * datum.test(i, qp);
  }

private:
  /// Coupled variable number
  unsigned int _v_var;
  /// Coupled variable
  GPUVariable _v;
  /// Multiplier for the coupled force term
  const Real _coef;
};
