//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details

#pragma once

#include "KokkosKernelBase.h"

namespace Moose::Kokkos
{

/**
 * Thread-local assembly data for a conforming interior face.
 */
class DGAssemblyDatum : public AssemblyDatum
{
public:
  KOKKOS_FUNCTION
  DGAssemblyDatum(const ContiguousElementID elem,
                  const unsigned int side,
                  const Assembly & assembly,
                  const Array<FESystem> & systems,
                  const Variable & ivar,
                  const unsigned int jvar)
    : AssemblyDatum(elem, side, assembly, systems, ivar, jvar),
      _neighbor_side(assembly.getNeighborSide(_elem, side)),
      _neighbor_reference_info(makeNeighborReferenceInfo(_neighbor, _elem.subdomain)),
      _n_neighbor_idofs(assembly.getNumDofs(_neighbor.type, _ife)),
      _n_neighbor_jdofs(systems[_sys].isScalarVariable(_jvar)
                            ? systems[_sys].getNumScalarDofs(_jvar)
                            : assembly.getNumDofs(_neighbor.type, _jfe))
  {
    KOKKOS_ASSERT(hasNeighbor());
  }

  KOKKOS_FUNCTION unsigned int neighborSide() const { return _neighbor_side; }

  KOKKOS_FUNCTION unsigned int neighborQp(const unsigned int qp) const
  {
    return _assembly.getNeighborQpIndex(_elem, _side, qp);
  }

  KOKKOS_FUNCTION const ElementInfo & neighborReferenceInfo() const
  {
    return _neighbor_reference_info;
  }

  KOKKOS_FUNCTION unsigned int nNeighborIDofs() const { return _n_neighbor_idofs; }
  KOKKOS_FUNCTION unsigned int nNeighborJDofs() const { return _n_neighbor_jdofs; }

  KOKKOS_FUNCTION Real elementVolume() const { return _mesh.getElementVolume(_elem.id); }
  KOKKOS_FUNCTION Real faceArea() const { return _mesh.getSideArea(_elem.id, _side); }

  KOKKOS_FUNCTION const Real33 & neighborJ(const unsigned int qp)
  {
    if (_cached_neighbor_qp != qp)
    {
      _assembly.computePhysicalMap(_neighbor_reference_info,
                                   _neighbor_side,
                                   neighborQp(qp),
                                   &_neighbor_J,
                                   nullptr,
                                   nullptr,
                                   nullptr);
      _cached_neighbor_qp = qp;
    }

    return _neighbor_J;
  }

private:
  static KOKKOS_FUNCTION ElementInfo
  makeNeighborReferenceInfo(ElementInfo neighbor, const ContiguousSubdomainID quadrature_subdomain)
  {
    neighbor.subdomain = quadrature_subdomain;
    return neighbor;
  }

  const unsigned int _neighbor_side;
  const ElementInfo _neighbor_reference_info;
  const unsigned int _n_neighbor_idofs;
  const unsigned int _n_neighbor_jdofs;
  unsigned int _cached_neighbor_qp = libMesh::invalid_uint;
  Real33 _neighbor_J;
};

template <bool is_test>
class NeighborVariableShapeValue
{
public:
  KOKKOS_FUNCTION Real
  operator()(DGAssemblyDatum & datum, const unsigned int i, const unsigned int qp) const
  {
    const auto fe = is_test ? datum.ife() : datum.jfe();
    return datum.assembly()
        .getPhiFace(datum.subdomain(), datum.neighbor().type, fe)(datum.neighborSide())(
            i, datum.neighborQp(qp));
  }
};

template <bool is_test>
class NeighborVariableShapeGradient
{
public:
  KOKKOS_FUNCTION const Real3 &
  reference(DGAssemblyDatum & datum, const unsigned int i, const unsigned int qp) const
  {
    const auto fe = is_test ? datum.ife() : datum.jfe();
    return datum.assembly()
        .getGradPhiFace(datum.subdomain(), datum.neighbor().type, fe)(datum.neighborSide())(
            i, datum.neighborQp(qp));
  }

  KOKKOS_FUNCTION Real3
  operator()(DGAssemblyDatum & datum, const unsigned int i, const unsigned int qp) const
  {
    return datum.neighborJ(qp) * reference(datum, i, qp);
  }
};

using NeighborVariablePhiValue = NeighborVariableShapeValue<false>;
using NeighborVariablePhiGradient = NeighborVariableShapeGradient<false>;
using NeighborVariableTestValue = NeighborVariableShapeValue<true>;
using NeighborVariableTestGradient = NeighborVariableShapeGradient<true>;

class NeighborVariableValue
{
public:
  NeighborVariableValue() = default;
  NeighborVariableValue(const MooseVariableFieldBase & var) : _var(var)
  {
    checkVariable(_var, false, "NeighborVariableValue");
  }

  KOKKOS_FUNCTION Real
  operator()(DGAssemblyDatum & datum, const unsigned int qp, const unsigned int comp = 0) const
  {
    KOKKOS_ASSERT(_var.initialized());

    if (!_var.coupled())
      return _var.value(_var.scalar() ? qp : comp);

    return datum.system(_var.sys(comp))
        .getVectorQpValueFace(datum.neighborReferenceInfo(),
                              datum.neighborSide(),
                              datum.neighborQp(qp),
                              _var.var(comp),
                              _var.tag());
  }

private:
  Variable _var;
};

class NeighborVariableGradient
{
public:
  NeighborVariableGradient() = default;
  NeighborVariableGradient(const MooseVariableFieldBase & var) : _var(var)
  {
    checkVariable(_var, false, "NeighborVariableGradient");
  }

  KOKKOS_FUNCTION Real3
  operator()(DGAssemblyDatum & datum, const unsigned int qp, const unsigned int comp = 0) const
  {
    KOKKOS_ASSERT(_var.initialized());

    if (!_var.coupled())
      return Real3(0);

    return datum.system(_var.sys(comp))
        .getVectorQpGradFace(datum.neighborReferenceInfo(),
                             datum.neighborSide(),
                             datum.neighborJ(qp),
                             datum.neighborQp(qp),
                             _var.var(comp),
                             _var.tag());
  }

private:
  Variable _var;
};

/**
 * Scalar, non-AD Kokkos DG kernel interface for conforming faces.
 */
class DGKernel : public KernelBase
{
public:
  static InputParameters validParams();

  static constexpr bool supports_scalar_jacobian = false;

  DGKernel(const InputParameters & parameters);

  virtual void computeResidual() override;
  virtual void computeJacobian() override;

  template <typename Derived>
  KOKKOS_FUNCTION Real computeQpJacobian(const Moose::DGJacobianType,
                                         const unsigned int,
                                         const unsigned int,
                                         const unsigned int,
                                         DGAssemblyDatum &) const
  {
    ::Kokkos::abort("Default computeQpJacobian() should never be called. Make sure you properly "
                    "redefined this method in your class without typos.");
    return 0;
  }

  template <typename Derived>
  KOKKOS_FUNCTION Real computeQpOffDiagJacobian(const Moose::DGJacobianType,
                                                const unsigned int,
                                                const unsigned int,
                                                const unsigned int,
                                                const unsigned int,
                                                DGAssemblyDatum &) const
  {
    ::Kokkos::abort(
        "Default computeQpOffDiagJacobian() should never be called. Make sure you properly "
        "redefined this method in your class without typos.");
    return 0;
  }

  template <typename Derived>
  static auto defaultJacobian()
  {
    return &DGKernel::computeQpJacobian<Derived>;
  }

  template <typename Derived>
  static auto defaultOffDiagJacobian()
  {
    return &DGKernel::computeQpOffDiagJacobian<Derived>;
  }

  template <typename Derived>
  KOKKOS_FUNCTION void operator()(ResidualLoop, const ThreadID tid, const Derived & kernel) const;

  template <typename Derived>
  KOKKOS_FUNCTION void operator()(JacobianLoop, const ThreadID tid, const Derived & kernel) const;

  template <typename Derived>
  KOKKOS_FUNCTION void
  operator()(OffDiagJacobianLoop, const ThreadID tid, const Derived & kernel) const;

  template <typename Derived>
  KOKKOS_FUNCTION void computeResidualInternal(const Derived & kernel,
                                               DGAssemblyDatum & datum,
                                               Moose::DGResidualType type) const;

  template <typename Derived>
  KOKKOS_FUNCTION void computeJacobianInternal(const Derived & kernel,
                                               DGAssemblyDatum & datum,
                                               Moose::DGJacobianType type) const;

  template <typename Derived>
  KOKKOS_FUNCTION void computeOffDiagJacobianInternal(const Derived & kernel,
                                                      DGAssemblyDatum & datum,
                                                      Moose::DGJacobianType type) const;

protected:
  const VariableTestValue _test;
  const VariableTestGradient _grad_test;
  const VariablePhiValue _phi;
  const VariablePhiGradient _grad_phi;
  const VariableValue _u;
  const VariableGradient _grad_u;

  const NeighborVariableTestValue _test_neighbor;
  const NeighborVariableTestGradient _grad_test_neighbor;
  const NeighborVariablePhiValue _phi_neighbor;
  const NeighborVariablePhiGradient _grad_phi_neighbor;
  const NeighborVariableValue _u_neighbor;
  const NeighborVariableGradient _grad_u_neighbor;

private:
  KOKKOS_FUNCTION unsigned int rowDofs(const DGAssemblyDatum & datum,
                                       Moose::DGJacobianType type) const;
  KOKKOS_FUNCTION unsigned int columnDofs(const DGAssemblyDatum & datum,
                                          Moose::DGJacobianType type) const;
  KOKKOS_FUNCTION ContiguousElementID rowElement(const DGAssemblyDatum & datum,
                                                 Moose::DGJacobianType type) const;
  KOKKOS_FUNCTION ContiguousElementID columnElement(const DGAssemblyDatum & datum,
                                                    Moose::DGJacobianType type) const;
  KOKKOS_FUNCTION bool activeBlock(const DGAssemblyDatum & datum,
                                   Moose::DGJacobianType type) const;
};

template <typename Derived>
KOKKOS_FUNCTION void
DGKernel::operator()(ResidualLoop, const ThreadID tid, const Derived & kernel) const
{
  const auto [elem, side] = kokkosBlockElementSideID(_thread(tid, 1));
  DGAssemblyDatum datum(
      elem, side, kokkosAssembly(), kokkosSystems(), _kokkos_var, _kokkos_var.var());
  datum.set_local_parallel(_thread(tid, 0), _thread.size(0));

  kernel.computeResidualInternal(kernel, datum, Moose::Element);
  kernel.computeResidualInternal(kernel, datum, Moose::Neighbor);
}

template <typename Derived>
KOKKOS_FUNCTION void
DGKernel::operator()(JacobianLoop, const ThreadID tid, const Derived & kernel) const
{
  const auto [elem, side] = kokkosBlockElementSideID(_thread(tid, 1));
  DGAssemblyDatum datum(
      elem, side, kokkosAssembly(), kokkosSystems(), _kokkos_var, _kokkos_var.var());
  datum.set_local_parallel(_thread(tid, 0), _thread.size(0));

  kernel.computeJacobianInternal(kernel, datum, Moose::ElementElement);
  kernel.computeJacobianInternal(kernel, datum, Moose::ElementNeighbor);
  kernel.computeJacobianInternal(kernel, datum, Moose::NeighborElement);
  kernel.computeJacobianInternal(kernel, datum, Moose::NeighborNeighbor);
}

template <typename Derived>
KOKKOS_FUNCTION void
DGKernel::operator()(OffDiagJacobianLoop, const ThreadID tid, const Derived & kernel) const
{
  const auto [elem, side] = kokkosBlockElementSideID(_thread(tid, 2));
  const auto & sys = kokkosSystem(_kokkos_var.sys());
  const auto jvar = sys.getFieldCoupling(_kokkos_var.var())[_thread(tid, 1)];

  DGAssemblyDatum datum(elem, side, kokkosAssembly(), kokkosSystems(), _kokkos_var, jvar);
  datum.set_local_parallel(_thread(tid, 0), _thread.size(0));

  kernel.computeOffDiagJacobianInternal(kernel, datum, Moose::ElementElement);
  kernel.computeOffDiagJacobianInternal(kernel, datum, Moose::ElementNeighbor);
  kernel.computeOffDiagJacobianInternal(kernel, datum, Moose::NeighborElement);
  kernel.computeOffDiagJacobianInternal(kernel, datum, Moose::NeighborNeighbor);
}

template <typename Derived>
KOKKOS_FUNCTION void
DGKernel::computeResidualInternal(const Derived & kernel,
                                  DGAssemblyDatum & datum,
                                  const Moose::DGResidualType type) const
{
  const bool neighbor_row = type == Moose::Neighbor;
  const auto & row_info = neighbor_row ? datum.neighbor() : datum.elem();
  const auto & sys = datum.system(datum.sys());
  if (!sys.isVariableActive(datum.ivar(), row_info.subdomain))
    return;

  const auto n_dofs = neighbor_row ? datum.nNeighborIDofs() : datum.n_idofs();
  Real local_re[MAX_CACHED_DOF];

  const unsigned int stride = MAX_CACHED_DOF * datum.num_local_threads();
  unsigned int num_batches = n_dofs / stride;
  if (n_dofs % stride)
    ++num_batches;

  for (unsigned int batch = 0; batch < num_batches; ++batch)
  {
    unsigned int ib = batch * stride;
    unsigned int ie = ::Kokkos::min(ib + stride, n_dofs);
    const unsigned int n = ie - ib;
    const unsigned int d = n / datum.num_local_threads();
    const unsigned int m = n % datum.num_local_threads();
    const unsigned int t = datum.local_thread_id();

    ib += t * d + (t < m ? t : m);
    ie = ib + d + (t < m ? 1 : 0);

    for (unsigned int i = ib; i < ie; ++i)
      local_re[i - ib] = 0;

    for (unsigned int qp = 0; qp < datum.n_qps(); ++qp)
      for (unsigned int i = ib; i < ie; ++i)
        local_re[i - ib] += datum.JxW(qp) *
                            kernel.template computeQpResidual<Derived>(type, i, qp, datum);

    for (unsigned int i = ib; i < ie; ++i)
      accumulateTaggedElementalResidual(local_re[i - ib], row_info.id, i);
  }
}

template <typename Derived>
KOKKOS_FUNCTION void
DGKernel::computeJacobianInternal(const Derived & kernel,
                                  DGAssemblyDatum & datum,
                                  const Moose::DGJacobianType type) const
{
  if (!activeBlock(datum, type))
    return;

  Real local_ke[MAX_CACHED_DOF];
  const auto n_rows = rowDofs(datum, type);
  const auto n_columns = columnDofs(datum, type);

  for (unsigned int j = datum.local_thread_id(); j < n_columns; j += datum.num_local_threads())
  {
    unsigned int num_batches = n_rows / MAX_CACHED_DOF;
    if (n_rows % MAX_CACHED_DOF)
      ++num_batches;

    for (unsigned int batch = 0; batch < num_batches; ++batch)
    {
      const unsigned int ib = batch * MAX_CACHED_DOF;
      const unsigned int ie = ::Kokkos::min(ib + MAX_CACHED_DOF, n_rows);

      for (unsigned int i = ib; i < ie; ++i)
        local_ke[i - ib] = 0;

      for (unsigned int qp = 0; qp < datum.n_qps(); ++qp)
        for (unsigned int i = ib; i < ie; ++i)
          local_ke[i - ib] += datum.JxW(qp) *
                              kernel.template computeQpJacobian<Derived>(type, i, j, qp, datum);

      for (unsigned int i = ib; i < ie; ++i)
        accumulateTaggedElementalMatrix(local_ke[i - ib],
                                        rowElement(datum, type),
                                        columnElement(datum, type),
                                        i,
                                        j,
                                        datum.jvar());
    }
  }
}

template <typename Derived>
KOKKOS_FUNCTION void
DGKernel::computeOffDiagJacobianInternal(const Derived & kernel,
                                         DGAssemblyDatum & datum,
                                         const Moose::DGJacobianType type) const
{
  if (!activeBlock(datum, type))
    return;

  Real local_ke[MAX_CACHED_DOF];
  const auto n_rows = rowDofs(datum, type);
  const auto n_columns = columnDofs(datum, type);

  for (unsigned int j = datum.local_thread_id(); j < n_columns; j += datum.num_local_threads())
  {
    unsigned int num_batches = n_rows / MAX_CACHED_DOF;
    if (n_rows % MAX_CACHED_DOF)
      ++num_batches;

    for (unsigned int batch = 0; batch < num_batches; ++batch)
    {
      const unsigned int ib = batch * MAX_CACHED_DOF;
      const unsigned int ie = ::Kokkos::min(ib + MAX_CACHED_DOF, n_rows);

      for (unsigned int i = ib; i < ie; ++i)
        local_ke[i - ib] = 0;

      for (unsigned int qp = 0; qp < datum.n_qps(); ++qp)
        for (unsigned int i = ib; i < ie; ++i)
          local_ke[i - ib] +=
              datum.JxW(qp) * kernel.template computeQpOffDiagJacobian<Derived>(
                                    type, i, j, datum.jvar(), qp, datum);

      for (unsigned int i = ib; i < ie; ++i)
        accumulateTaggedElementalMatrix(local_ke[i - ib],
                                        rowElement(datum, type),
                                        columnElement(datum, type),
                                        i,
                                        j,
                                        datum.jvar());
    }
  }
}

KOKKOS_FUNCTION inline unsigned int
DGKernel::rowDofs(const DGAssemblyDatum & datum, const Moose::DGJacobianType type) const
{
  return type == Moose::ElementElement || type == Moose::ElementNeighbor ? datum.n_idofs()
                                                                         : datum.nNeighborIDofs();
}

KOKKOS_FUNCTION inline unsigned int
DGKernel::columnDofs(const DGAssemblyDatum & datum, const Moose::DGJacobianType type) const
{
  return type == Moose::ElementElement || type == Moose::NeighborElement ? datum.n_jdofs()
                                                                         : datum.nNeighborJDofs();
}

KOKKOS_FUNCTION inline ContiguousElementID
DGKernel::rowElement(const DGAssemblyDatum & datum, const Moose::DGJacobianType type) const
{
  return type == Moose::ElementElement || type == Moose::ElementNeighbor ? datum.elemID()
                                                                         : datum.neighborID();
}

KOKKOS_FUNCTION inline ContiguousElementID
DGKernel::columnElement(const DGAssemblyDatum & datum, const Moose::DGJacobianType type) const
{
  return type == Moose::ElementElement || type == Moose::NeighborElement ? datum.elemID()
                                                                         : datum.neighborID();
}

KOKKOS_FUNCTION inline bool
DGKernel::activeBlock(const DGAssemblyDatum & datum, const Moose::DGJacobianType type) const
{
  const auto & sys = datum.system(datum.sys());
  const auto row_subdomain =
      type == Moose::ElementElement || type == Moose::ElementNeighbor ? datum.subdomain()
                                                                      : datum.neighborSubdomain();
  const auto column_subdomain =
      type == Moose::ElementElement || type == Moose::NeighborElement ? datum.subdomain()
                                                                      : datum.neighborSubdomain();

  return sys.isVariableActive(datum.ivar(), row_subdomain) &&
         sys.isVariableActive(datum.jvar(), column_subdomain);
}

} // namespace Moose::Kokkos

using DGAssemblyDatum = Moose::Kokkos::DGAssemblyDatum;
