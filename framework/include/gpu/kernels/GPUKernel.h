//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "GPUKernelBase.h"

#include "MooseVariableInterface.h"

template <typename Kernel>
class GPUKernel : public GPUKernelBase, public MooseVariableInterface<Real>
{
public:
  static InputParameters validParams()
  {
    InputParameters params = GPUKernelBase::validParams();
    params.registerBase("Kernel");
    return params;
  }

  // Constructor
  GPUKernel(const InputParameters & parameters)
    : GPUKernelBase(parameters),
      MooseVariableInterface<Real>(this,
                                   false,
                                   "variable",
                                   Moose::VarKindType::VAR_SOLVER,
                                   Moose::VarFieldType::VAR_FIELD_STANDARD),
      _var(*mooseVariable()),
      _default_diag(&Kernel::computeQpJacobian == &GPUKernel::computeQpJacobian),
      _default_offdiag(&Kernel::computeQpOffDiagJacobian == &GPUKernel::computeQpOffDiagJacobian)
  {
    addMooseVariableDependency(mooseVariable());

    // Tells the variable to GPU
    setVariable(_var);
  }

  // Copy constructor
  GPUKernel(const GPUKernel<Kernel> & object)
    : GPUKernelBase(object),
      MooseVariableInterface<Real>(this,
                                   false,
                                   "variable",
                                   Moose::VarKindType::VAR_SOLVER,
                                   Moose::VarFieldType::VAR_FIELD_STANDARD,
                                   false),
      _var(object._var),
      _default_diag(object._default_diag),
      _default_offdiag(object._default_offdiag)
  {
    _thread = object._thread;
  }

  virtual const MooseVariable & variable() const override { return _var; }

  // Dispatch residual calculation to GPU
  virtual void computeResidual() override
  {
    Kokkos::RangePolicy<ResidualLoop, Kokkos::IndexType<size_t>> policy(0, numBlockElements());
    Kokkos::parallel_for(policy, *static_cast<Kernel *>(this));
  }
  // Dispatch Jacobian calculation to GPU
  virtual void computeJacobian() override
  {
    auto & system = systems()[_gpu_var.sys()];

    _thread.resize({system.getCoupling(_gpu_var.var()).size(), numBlockElements()});

    Kokkos::RangePolicy<JacobianLoop, Kokkos::IndexType<size_t>> policy(0, _thread.size());
    Kokkos::parallel_for(policy, *static_cast<Kernel *>(this));
  }

  // Empty methods to prevent compile errors even when these methods were not hidden by the derived
  // class
  KOKKOS_FUNCTION Real computeQpJacobian(const unsigned int /* i */,
                                         const unsigned int /* j */,
                                         const unsigned int /* qp */,
                                         const ResidualDatum & /* datum */) const
  {
    return 0;
  }
  KOKKOS_FUNCTION Real computeQpOffDiagJacobian(const unsigned int /* i */,
                                                const unsigned int /* j */,
                                                const unsigned int /* jvar */,
                                                const unsigned int /* qp */,
                                                const ResidualDatum & /* datum */) const
  {
    return 0;
  }

  // Overloaded operators called by Kokkos::parallel_for
  KOKKOS_FUNCTION void operator()(ResidualLoop, const size_t tid) const
  {
    auto kernel = static_cast<const Kernel *>(this);
    auto elem = blockElementID(tid);

    ResidualDatum datum(elem, assembly(), systems(), _gpu_var, _gpu_var.var());

    Real local_re[MAX_DOF];

    for (unsigned int i = 0; i < datum.n_dofs(); ++i)
      local_re[i] = 0;

    for (unsigned int qp = 0; qp < datum.n_qps(); ++qp)
    {
      datum.prefetch(qp);

      for (unsigned int i = 0; i < datum.n_dofs(); ++i)
        local_re[i] += datum.JxWCoord(qp) * kernel->computeQpResidual(i, qp, datum);
    }

    for (unsigned int i = 0; i < datum.n_dofs(); ++i)
      accumulateTaggedLocalResidual(local_re[i], elem, i);
  }
  KOKKOS_FUNCTION void operator()(JacobianLoop, const size_t tid) const
  {
    auto & system = systems()[_gpu_var.sys()];
    auto jvar = system.getCoupling(_gpu_var.var())[_thread(tid, 0)];
    auto diag = _gpu_var.var() == jvar;

    if ((diag && _default_diag) || (!diag && _default_offdiag))
      return;

    auto kernel = static_cast<const Kernel *>(this);
    auto elem = blockElementID(_thread(tid, 1));

    if (!system.isVariableActive(jvar, assembly().getElementInfo(elem).subdomain))
      return;

    ResidualDatum datum(elem, assembly(), systems(), _gpu_var, jvar);

    Real local_ke[MAX_DOF];

    for (unsigned int i = 0; i < datum.n_idofs(); ++i)
    {
      for (unsigned int j = 0; j < datum.n_jdofs(); ++j)
        local_ke[j] = 0;

      for (unsigned int qp = 0; qp < datum.n_qps(); ++qp)
      {
        datum.prefetch(qp);

        for (unsigned int j = 0; j < datum.n_jdofs(); ++j)
          local_ke[j] +=
              datum.JxWCoord(qp) * (diag ? kernel->computeQpJacobian(i, j, qp, datum)
                                         : kernel->computeQpOffDiagJacobian(i, j, jvar, qp, datum));
      }

      for (unsigned int j = 0; j < datum.n_jdofs(); ++j)
        accumulateTaggedLocalMatrix(local_ke[j], elem, i, j, jvar);
    }
  }

protected:
  // Reference to MooseVariable
  MooseVariable & _var;

private:
  // Whether default computeQpJacobian is used
  const bool _default_diag;
  // Whether default computeQpOffDiagJacobian is used
  const bool _default_offdiag;
  // GPU thread object
  GPUThread _thread;
};
