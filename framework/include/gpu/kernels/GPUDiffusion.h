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
 * This kernel implements the Laplacian operator:
 * $\nabla u \cdot \nabla \phi_i$
 */
template <typename Kernel>
class GPUDiffusion : public GPUKernel<Kernel>
{
public:
  static InputParameters validParams()
  {
    InputParameters params = GPUKernel<Kernel>::validParams();
    return params;
  }

  GPUDiffusion(const InputParameters & parameters) : GPUKernel<Kernel>(parameters) {}

  KOKKOS_FUNCTION Real computeQpResidual(const unsigned int i,
                                         const unsigned int qp,
                                         const ResidualDatum & datum) const
  {
    return datum.grad_u(qp) * datum.grad_test(i, qp);
  }
  KOKKOS_FUNCTION Real computeQpJacobian(const unsigned int i,
                                         const unsigned int j,
                                         const unsigned int qp,
                                         const ResidualDatum & datum) const
  {
    return datum.grad_phi(j, qp) * datum.grad_test(i, qp);
  }
};

class GPUDiffusionKernel final : public GPUDiffusion<GPUDiffusionKernel>
{
public:
  static InputParameters validParams();

  GPUDiffusionKernel(const InputParameters & parameters);
};
