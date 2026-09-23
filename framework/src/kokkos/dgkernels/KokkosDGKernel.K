//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details

#include "KokkosDGKernel.h"

namespace Moose::Kokkos
{

InputParameters
DGKernel::validParams()
{
  auto params = KernelBase::validParams();

  params.set<Moose::MaterialDataType>("_material_data_type") = Moose::BOUNDARY_MATERIAL_DATA;

  params.addRelationshipManager("ElementSideNeighborLayers",
                                Moose::RelationshipManagerType::GEOMETRIC |
                                    Moose::RelationshipManagerType::ALGEBRAIC |
                                    Moose::RelationshipManagerType::COUPLING);

  params.addPrivateParam<bool>("_kokkos_dg_kernel", true);
  params.registerBase("DGKernel");

  return params;
}

DGKernel::DGKernel(const InputParameters & parameters)
  : KernelBase(parameters, Moose::VarFieldType::VAR_FIELD_STANDARD),
    _test(),
    _grad_test(),
    _phi(),
    _grad_phi(),
    _u(_var),
    _grad_u(_var),
    _test_neighbor(),
    _grad_test_neighbor(),
    _phi_neighbor(),
    _grad_phi_neighbor(),
    _u_neighbor(_var),
    _grad_u_neighbor(_var)
{
  addMooseVariableDependency(&_var);
}

void
DGKernel::computeResidual()
{
  _thread.resize(_num_local_residual_threads, numKokkosBlockSides());
  Policy policy(0, _thread.size());

  if (!_residual_dispatcher)
    _residual_dispatcher = DispatcherRegistry::build<ResidualLoop>(this, type());

  _residual_dispatcher->parallelFor(policy);
}

void
DGKernel::computeJacobian()
{
  if (DispatcherRegistry::hasUserMethod<JacobianLoop>(type()))
  {
    _thread.resize(_num_local_jacobian_threads, numKokkosBlockSides());
    Policy policy(0, _thread.size());

    if (!_jacobian_dispatcher)
      _jacobian_dispatcher = DispatcherRegistry::build<JacobianLoop>(this, type());

    _jacobian_dispatcher->parallelFor(policy);
  }

  if (DispatcherRegistry::hasUserMethod<OffDiagJacobianLoop>(type()))
  {
    auto & sys = kokkosSystem(_kokkos_var.sys());
    _thread.resize(_num_local_jacobian_threads,
                   sys.getFieldCoupling(_kokkos_var.var()).size(),
                   numKokkosBlockSides());
    Policy policy(0, _thread.size());

    if (!_offdiag_jacobian_dispatcher)
      _offdiag_jacobian_dispatcher = DispatcherRegistry::build<OffDiagJacobianLoop>(this, type());

    _offdiag_jacobian_dispatcher->parallelFor(policy);
  }
}

} // namespace Moose::Kokkos
