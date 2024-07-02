//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "GPUNodalBCBase.h"

#include "MooseVariableInterface.h"

template <typename NodalBC>
class GPUNodalBC : public GPUNodalBCBase, public MooseVariableInterface<Real>
{
public:
  static InputParameters validParams()
  {
    InputParameters params = GPUNodalBCBase::validParams();

    return params;
  }

  GPUNodalBC(const InputParameters & parameters)
    : GPUNodalBCBase(parameters),
      MooseVariableInterface<Real>(this,
                                   true,
                                   "variable",
                                   Moose::VarKindType::VAR_SOLVER,
                                   Moose::VarFieldType::VAR_FIELD_STANDARD),
      _var(*mooseVariable()),
      _default_offdiag(&NodalBC::computeQpOffDiagJacobian == &GPUNodalBC::computeQpOffDiagJacobian)
  {
    addMooseVariableDependency(mooseVariable());

    // Tells the variable to GPU
    setVariable(_var);
  }

  GPUNodalBC(const GPUNodalBC<NodalBC> & object)
    : GPUNodalBCBase(object),
      MooseVariableInterface<Real>(this,
                                   true,
                                   "variable",
                                   Moose::VarKindType::VAR_SOLVER,
                                   Moose::VarFieldType::VAR_FIELD_STANDARD,
                                   false),
      _var(object._var),
      _default_offdiag(object._default_offdiag)
  {
    _thread = object._thread;
  }

  virtual const MooseVariable & variable() const override { return _var; }

  // Dispatch residual calculation to GPU
  virtual void computeResidual() override
  {
    Kokkos::RangePolicy<ResidualLoop, Kokkos::IndexType<size_t>> policy(0, numBoundaryNodes());
    Kokkos::parallel_for(policy, *static_cast<NodalBC *>(this));

    Kokkos::fence();
  }
  // Dispatch diagonal Jacobian calculation to GPU
  virtual void computeJacobian() override
  {
    auto & system = systems()[_gpu_var.sys()];

    _thread.resize({system.getCoupling(_gpu_var.var()).size(), numBoundaryNodes()});

    Kokkos::RangePolicy<JacobianLoop, Kokkos::IndexType<size_t>> policy(0, _thread.size());
    Kokkos::parallel_for(policy, *static_cast<NodalBC *>(this));

    Kokkos::fence();
  }

  // Empty methods to prevent compile errors even when these methods were not hidden by the derived
  // class
  KOKKOS_FUNCTION Real computeQpJacobian(const ResidualNodalDatum & /* datum */) const { return 1; }
  KOKKOS_FUNCTION Real computeQpOffDiagJacobian(const unsigned int /* jvar */,
                                                const ResidualNodalDatum & /* datum */) const
  {
    return 0;
  }

  // Overloaded operators called by Kokkos::parallel_for
  KOKKOS_FUNCTION void operator()(ResidualLoop, const size_t tid) const
  {
    auto bc = static_cast<const NodalBC *>(this);
    auto node = boundaryNodeID(tid);

    ResidualNodalDatum datum(node, systems(), _gpu_var);

    Real local_re = bc->computeQpResidual(datum);

    setTaggedLocalResidual(local_re, node);
  }
  KOKKOS_FUNCTION void operator()(JacobianLoop, const size_t tid) const
  {
    auto & system = systems()[_gpu_var.sys()];
    auto jvar = system.getCoupling(_gpu_var.var())[_thread(tid, 0)];
    auto diag = _gpu_var.var() == jvar;

    if (!diag && _default_offdiag)
      return;

    auto bc = static_cast<const NodalBC *>(this);
    auto node = boundaryNodeID(_thread(tid, 1));

    ResidualNodalDatum datum(node, systems(), _gpu_var);

    Real local_ke = diag ? bc->computeQpJacobian(datum) : bc->computeQpOffDiagJacobian(jvar, datum);

    setTaggedLocalMatrix(local_ke, node, jvar);
  }

protected:
  // Reference to MooseVariable
  MooseVariable & _var;

private:
  // Whether default computeQpOffDiagJacobian is used
  const bool _default_offdiag;
  // GPU thread object
  GPUThread _thread;
};
