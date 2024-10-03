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

template <typename Kernel>
class GPUTimeKernel : public GPUKernel<Kernel>
{
public:
  static InputParameters validParams()
  {
    InputParameters params = GPUKernel<Kernel>::validParams();

    params.set<MultiMooseEnum>("vector_tags") = "time";
    params.set<MultiMooseEnum>("matrix_tags") = "system time";

    return params;
  }

  // Constructor
  GPUTimeKernel(const InputParameters & parameters) : GPUKernel<Kernel>(parameters) {}

  // Empty method to prevent compile errors even when this method was not hidden by the derived
  // class
  KOKKOS_FUNCTION void computeResidualAdditional(Real * /* local_re */,
                                                 const ResidualDatum & /* datum */) const
  {
  }

  // Overloaded operator called by Kokkos::parallel_for
  KOKKOS_FUNCTION void operator()(ResidualLoop, const size_t tid) const
  {
    auto kernel = static_cast<const Kernel *>(this);
    auto elem = this->blockElementID(tid);

    ResidualDatum datum(elem, this->assembly(), this->systems(), _gpu_var, _gpu_var.var());

    Real local_re[MAX_DOF];

    for (unsigned int i = 0; i < datum.n_dofs(); ++i)
      local_re[i] = 0;

    for (unsigned int qp = 0; qp < datum.n_qps(); ++qp)
    {
      datum.prefetch(qp);

      for (unsigned int i = 0; i < datum.n_dofs(); ++i)
        local_re[i] += datum.JxWCoord(qp) * kernel->computeQpResidual(i, qp, datum);
    }

    kernel->computeResidualAdditional(local_re, datum);

    for (unsigned int i = 0; i < datum.n_dofs(); ++i)
      this->accumulateTaggedLocalResidual(local_re[i], elem, i);
  }
};
