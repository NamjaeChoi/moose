//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "GPUTypes.h"
#include "GPUAssembly.h"
#include "GPUSystem.h"
#include "GPUMaterialProperty.h"

class Datum
{
public:
  KOKKOS_FUNCTION
  Datum(const dof_id_type elem,
        const unsigned int side,
        const GPUAssembly & assembly,
        const GPUArray<GPUSystem> & systems)
    : _assembly(assembly),
      _systems(systems),
      _elem(assembly.getElementInfo(elem)),
      _side(side),
      _neighbor(_side != -1 ? assembly.getNeighbor(_elem.id, side) : -1),
      _n_qps(side == -1 ? assembly.n_qps(_elem) : assembly.n_qps_face(_elem, side)),
      _qp_offset(side == -1 ? assembly.getQpOffset(_elem) : assembly.getQpFaceOffset(_elem, side)),
      _Jac(side == -1 ? assembly.getJacobian(_elem.subdomain)
                      : assembly.getJacobianFace(_elem.subdomain)),
      _JxW_coord(side == -1 ? assembly.getJxWCoord(_elem.subdomain)
                            : assembly.getJxWCoordFace(_elem.subdomain))
  {
  }
  KOKKOS_FUNCTION
  Datum(const dof_id_type elem, const GPUAssembly & assembly, const GPUArray<GPUSystem> & systems)
    : Datum(elem, -1, assembly, systems)
  {
  }

  KOKKOS_FUNCTION bool hasNeighbor() { return _neighbor != -1; }

  KOKKOS_FUNCTION unsigned int n_qps() const { return _n_qps; }
  KOKKOS_FUNCTION unsigned int n_dofs(const unsigned int fe) const
  {
    return _assembly.n_dofs(_elem.id, fe);
  }

  // Coupleable
  KOKKOS_FUNCTION Real coupledValue(const GPUVariable & var,
                                    const unsigned int qp,
                                    const unsigned int comp = 0) const
  {
    if (var.coupled())
      return _side == -1 ? _systems[var.sys()].getVectorQpValue(
                               _elem, _qp_offset + qp, var.var(comp), var.tag())
                         : _systems[var.sys()].getVectorQpValueFace(
                               _elem, _side, qp, var.var(comp), var.tag());
    else
      return var.value(comp);
  }
  KOKKOS_FUNCTION Real3 coupledGradient(const GPUVariable & var,
                                        const unsigned int qp,
                                        const unsigned int comp = 0) const
  {
    if (var.coupled())
      return _side == -1 ? _systems[var.sys()].getVectorQpGrad(
                               _elem, _qp_offset + qp, var.var(comp), var.tag())
                         : _systems[var.sys()].getVectorQpGradFace(
                               _elem, _side, _qp_offset + qp, qp, var.var(comp), var.tag());
    else
      return Real3(0);
  }
  KOKKOS_FUNCTION Real coupledDofValue(const GPUVariable & var,
                                       const unsigned int i,
                                       const unsigned int comp = 0) const
  {
    if (var.coupled())
    {
      auto dof = _systems[var.sys()].getElemLocalDofIndex(_elem.id, i, var.var(comp));
      return _systems[var.sys()].getVectorDofValue(dof, var.tag());
    }
    else
      return var.value(comp);
  }

  // Material property
  template <typename T, unsigned int dimension>
  KOKKOS_FUNCTION const GPUProperty<T, dimension>
  getProperty(const GPUMaterialProperty<T, dimension> & prop, const unsigned int qp) const
  {
    return GPUProperty<T, dimension>(prop, _elem.subdomain, _qp_offset + qp);
  }
  template <typename T, unsigned int dimension>
  KOKKOS_FUNCTION GPUProperty<T, dimension>
  setProperty(const GPUMaterialProperty<T, dimension> & prop, const unsigned int qp) const
  {
    return GPUProperty<T, dimension>(prop, _elem.subdomain, _qp_offset + qp);
  }

  // Assembly
  KOKKOS_FUNCTION Real phi(const unsigned int fe,
                           const unsigned int dof,
                           const unsigned int qp) const
  {
    return _side == -1 ? _assembly.getPhi(_elem.subdomain, _elem.type, fe)(dof, qp)
                       : _assembly.getPhiFace(_elem.subdomain, _elem.type, fe)(dof, qp, _side);
  }
  KOKKOS_FUNCTION Real3 grad_phi(const unsigned int fe,
                                 const unsigned int dof,
                                 const unsigned int qp) const
  {
    return Jac(_qp_offset + qp) *
           (_side == -1
                ? _assembly.getGradPhi(_elem.subdomain, _elem.type, fe)(dof, qp)
                : _assembly.getGradPhiFace(_elem.subdomain, _elem.type, fe)(dof, qp, _side));
  }
  KOKKOS_FUNCTION Real33 Jac(const unsigned int qp) const
  {
    return _prefetched ? _Jac_prefetch : _Jac(_qp_offset + qp);
  }
  KOKKOS_FUNCTION Real JxWCoord(const unsigned int qp) const
  {
    return _prefetched ? _JxW_prefetch : _JxW_coord(_qp_offset + qp);
  }
  KOKKOS_FUNCTION Real prefetch(const unsigned int qp)
  {
    _Jac_prefetch = _Jac(_qp_offset + qp);
    _JxW_prefetch = _JxW_coord(_qp_offset + qp);
    _prefetched = true;
  }

protected:
  // Reference to the GPU assembly
  const GPUAssembly & _assembly;
  // Reference to the GPU system
  const GPUArray<GPUSystem> & _systems;
  // Element information
  const GPUElementInfo _elem;
  // Side index
  const unsigned int _side;
  // Neighbor local element ID
  const dof_id_type _neighbor;
  // Number of quadrature points
  const unsigned int _n_qps;
  // Quadrature point offset
  const dof_id_type _qp_offset;
  // Reference to transformation data
  const GPUArray<Real33> & _Jac;
  const GPUArray<Real> & _JxW_coord;
  // Prefetched data
  Real33 _Jac_prefetch;
  Real _JxW_prefetch;
  bool _prefetched = false;
};

class ResidualDatum : public Datum
{
public:
  KOKKOS_FUNCTION
  ResidualDatum(const dof_id_type elem,
                const unsigned int side,
                const GPUAssembly & assembly,
                const GPUArray<GPUSystem> & systems,
                const GPUVariable & ivar,
                const unsigned int jvar,
                const unsigned int comp = 0)
    : Datum(elem, side, assembly, systems),
      _system(systems[ivar.sys()]),
      _tag(ivar.tag()),
      _ivar(ivar.var(comp)),
      _jvar(jvar),
      _ife(_system.getFETypeNum(_ivar)),
      _jfe(_system.getFETypeNum(_jvar)),
      _n_idofs(_assembly.n_dofs(_elem.id, _ife)),
      _n_jdofs(_assembly.n_dofs(_elem.id, _jfe))
  {
  }
  KOKKOS_FUNCTION
  ResidualDatum(const dof_id_type elem,
                const GPUAssembly & assembly,
                const GPUArray<GPUSystem> & systems,
                const GPUVariable & ivar,
                const unsigned int jvar,
                const unsigned int comp = 0)
    : ResidualDatum(elem, -1, assembly, systems, ivar, jvar, comp)
  {
  }

  KOKKOS_FUNCTION unsigned int n_dofs() const { return _n_idofs; }
  KOKKOS_FUNCTION unsigned int n_idofs() const { return _n_idofs; }
  KOKKOS_FUNCTION unsigned int n_jdofs() const { return _n_jdofs; }

  // Solution
  KOKKOS_FUNCTION Real u(const unsigned int qp) const
  {
    return _side == -1 ? _system.getVectorQpValue(_elem, _qp_offset + qp, _ivar, _tag)
                       : _system.getVectorQpValueFace(_elem, _side, qp, _ivar, _tag);
  }
  KOKKOS_FUNCTION Real3 grad_u(const unsigned int qp) const
  {
    return _side == -1
               ? _system.getVectorQpGrad(_elem, _qp_offset + qp, _ivar, _tag)
               : _system.getVectorQpGradFace(_elem, _side, _qp_offset + qp, qp, _ivar, _tag);
  }

  // Assembly
  KOKKOS_FUNCTION Real phi(const unsigned int dof, const unsigned int qp) const
  {
    return Datum::phi(_jfe, dof, qp);
  }
  KOKKOS_FUNCTION Real3 grad_phi(const unsigned int dof, const unsigned int qp) const
  {
    return Datum::grad_phi(_jfe, dof, qp);
  }
  KOKKOS_FUNCTION Real test(const unsigned int dof, const unsigned int qp) const
  {
    return Datum::phi(_ife, dof, qp);
  }
  KOKKOS_FUNCTION Real3 grad_test(const unsigned int dof, const unsigned int qp) const
  {
    return Datum::grad_phi(_ife, dof, qp);
  }

protected:
  // Solution tag ID
  const TagID _tag;
  // Reference to the solution system
  const GPUSystem & _system;
  // Variable numbers
  const unsigned int _ivar, _jvar;
  // FE type numbers of variables
  const unsigned int _ife, _jfe;
  // Number of DOFs
  const unsigned int _n_idofs, _n_jdofs;
};

class NeighborDatum : public Datum
{
public:
  KOKKOS_FUNCTION
  NeighborDatum(const dof_id_type elem,
                const unsigned int side,
                const GPUAssembly & assembly,
                const GPUArray<GPUSystem> & systems)
    : Datum(elem, side, assembly, systems)
  {
  }

  // Coupleable
  KOKKOS_FUNCTION Real coupledNeighborValue(const GPUVariable & var,
                                            const unsigned int qp,
                                            const unsigned int comp = 0) const
  {
    if (var.coupled())
      return _systems[var.sys()].getVectorQpValueNeighbor(
          _elem, _neighbor, _qp_offset + qp, var.var(comp), var.tag());
    else
      return var.value(comp);
  }
  KOKKOS_FUNCTION Real3 coupledNeighborGradient(const GPUVariable & var,
                                                const unsigned int qp,
                                                const unsigned int comp = 0) const
  {
    if (var.coupled())
      return _systems[var.sys()].getVectorQpGradNeighbor(
          _elem, _neighbor, _qp_offset + qp, var.var(comp), var.tag());
    else
      return Real3(0);
  }

  // Assembly
  KOKKOS_FUNCTION Real phi_neighbor(const unsigned int fe,
                                    const unsigned int dof,
                                    const unsigned int qp) const
  {
    return _assembly.getPhiNeighbor(_elem.subdomain, fe)(dof, _qp_offset + qp);
  }
  KOKKOS_FUNCTION Real3 grad_phi_neighbor(const unsigned int fe,
                                          const unsigned int dof,
                                          const unsigned int qp) const
  {
    return _assembly.getGradPhiNeighbor(_elem.subdomain, fe)(dof, _qp_offset + qp);
  }
};

class NodalDatum
{
public:
  KOKKOS_FUNCTION
  NodalDatum(const dof_id_type node, const GPUArray<GPUSystem> & systems)
    : _systems(systems), _node(node)
  {
  }

  // Coupleable
  KOKKOS_FUNCTION Real coupledNodalValue(const GPUVariable & var, const unsigned int comp = 0) const
  {
    if (var.coupled())
    {
      auto dof = _systems[var.sys()].getNodeLocalDofIndex(_node, var.var(comp));
      return _systems[var.sys()].getVectorDofValue(dof, var.tag());
    }
    else
      return var.value(comp);
  }

protected:
  // Reference to the GPU system
  const GPUArray<GPUSystem> & _systems;
  // Local node ID
  const dof_id_type _node;
};

class ResidualNodalDatum : public NodalDatum
{
public:
  KOKKOS_FUNCTION
  ResidualNodalDatum(const dof_id_type node,
                     const GPUArray<GPUSystem> & systems,
                     const GPUVariable & var,
                     const unsigned int comp = 0)
    : NodalDatum(node, systems),
      _var(var.var(comp)),
      _tag(var.tag()),
      _system(systems[var.sys()]),
      _dof(_system.getNodeLocalDofIndex(_node, _var))
  {
  }

  // Solution
  KOKKOS_FUNCTION Real u() const { return _system.getVectorDofValue(_dof, _tag); }

private:
  // Variable number
  const unsigned int _var;
  // Solution tag ID
  const TagID _tag;
  // Reference to the solution system
  const GPUSystem & _system;
  // Local DOF index
  const dof_id_type _dof;
};
