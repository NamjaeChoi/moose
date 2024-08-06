//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "GPUDatum.h"

#include "MooseVariableBase.h"
#include "ResidualObject.h"

class GPUResidualObject : public ResidualObject
{
public:
  static InputParameters validParams();

  GPUResidualObject(const InputParameters & parameters, bool nodal = false);
  GPUResidualObject(const GPUResidualObject & object);

  // GPU function tags
  struct ResidualLoop
  {
  };
  struct JacobianLoop
  {
  };

  virtual void computeOffDiagJacobian(unsigned int) override final
  {
    mooseError("computeOffDiagJacobian() is not used for GPU residual objects.");
  }
  virtual void computeResidualAndJacobian() override final
  {
    computeResidual();
    computeJacobian();
  }

protected:
  // GPU variable this object operates on
  GPUVariable _gpu_var;

private:
  // Tags this object operates on
  GPUArray<TagID> _vector_tags;
  GPUArray<TagID> _matrix_tags;
  // Copy of GPU assembly (for device access)
  GPUAssembly _assembly;
  // Reference to GPU assembly (for host access)
  GPUAssembly & _assembly_ref;
  // Copy of GPU systems (for device access)
  GPUArray<GPUSystem> _systems;
  // Reference to GPU systems (for host access)
  GPUArray<GPUSystem> & _systems_ref;

protected:
  KOKKOS_FUNCTION const GPUAssembly & assembly() const
  {
    KOKKOS_IF_ON_HOST(return _assembly_ref;)
    KOKKOS_IF_ON_DEVICE(return _assembly;)
  }
  KOKKOS_FUNCTION const GPUArray<GPUSystem> & systems() const
  {
    KOKKOS_IF_ON_HOST(return _systems_ref;)
    KOKKOS_IF_ON_DEVICE(return _systems;)
  }

protected:
  // Sets the GPU variable this object operates on
  void setVariable(const MooseVariableBase & var);
  // Accumulate local residual to tagged vectors
  KOKKOS_FUNCTION void accumulateTaggedLocalResidual(Real local_re,
                                                     dof_id_type elem,
                                                     unsigned int i,
                                                     unsigned int comp = 0) const
  {
    if (!local_re)
      return;

    auto & system = systems()[_gpu_var._sys];
    auto dof = system.getElemLocalDofIndex(elem, i, _gpu_var._vars[comp]);

    for (size_t t = 0; t < _vector_tags.size(); ++t)
    {
      auto tag = _vector_tags[t];

      if (system.isResidualTagActive(tag) && !system.hasNodalResidual(dof, tag))
        Kokkos::atomic_add(&system.getVectorDofValue(dof, tag), local_re);
    }
  }
  // Set local residual to tagged vectors
  KOKKOS_FUNCTION void
  setTaggedLocalResidual(Real local_re, dof_id_type node, unsigned int comp = 0) const
  {
    if (!local_re)
      return;

    auto & system = systems()[_gpu_var._sys];
    auto dof = system.getNodeLocalDofIndex(node, _gpu_var._vars[comp]);

    for (size_t t = 0; t < _vector_tags.size(); ++t)
    {
      auto tag = _vector_tags[t];

      if (system.isResidualTagActive(tag))
        system.getVectorDofValue(dof, tag) = local_re;
    }
  }
  // Accumulate local Jacobian to tagged matrices
  KOKKOS_FUNCTION void accumulateTaggedLocalMatrix(Real local_ke,
                                                   dof_id_type elem,
                                                   unsigned int i,
                                                   unsigned int j,
                                                   unsigned int jvar,
                                                   unsigned int comp = 0) const
  {
    if (!local_ke)
      return;

    auto & system = systems()[_gpu_var._sys];
    auto row = system.getElemLocalDofIndex(elem, i, _gpu_var._vars[comp]);
    auto col = system.getElemGlobalDofIndex(elem, j, jvar);

    for (size_t t = 0; t < _matrix_tags.size(); ++t)
    {
      auto tag = _matrix_tags[t];

      if (system.isMatrixTagActive(tag) && !system.hasNodalJacobian(row, tag))
        Kokkos::atomic_add(&system.getMatrixDofValue(row, col, tag), local_ke);
    }
  }
  // Set local Jacobian to tagged matrices
  KOKKOS_FUNCTION void setTaggedLocalMatrix(Real local_ke,
                                            dof_id_type node,
                                            unsigned int jvar,
                                            unsigned int comp = 0) const
  {
    if (!local_ke)
      return;

    auto & system = systems()[_gpu_var._sys];
    auto row = system.getNodeLocalDofIndex(node, _gpu_var._vars[comp]);
    auto col = system.getNodeGlobalDofIndex(node, jvar);

    for (size_t t = 0; t < _matrix_tags.size(); ++t)
    {
      auto tag = _matrix_tags[t];

      if (system.isMatrixTagActive(tag))
      {
        auto & matrix = system.getMatrix(tag);

        matrix.zero(row);
        matrix(row, col) = local_ke;
      }
    }
  }
};
