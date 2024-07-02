//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "GPUNodalBC.h"

template <typename DirichletBC>
class GPUDirichletBCBase : public GPUNodalBC<DirichletBC>
{
public:
  static InputParameters validParams()
  {
    InputParameters params = GPUNodalBC<DirichletBC>::validParams();
    params.addParam<bool>(
        "preset", true, "Whether or not to preset the BC (apply the value before the solve begins).");
    return params;
  }

  GPUDirichletBCBase(const InputParameters & parameters)
    : GPUNodalBC<DirichletBC>(parameters), _preset(this->template getParam<bool>("preset"))
  {
  }

  GPUDirichletBCBase(const GPUDirichletBCBase<DirichletBC> & object)
    : GPUNodalBC<DirichletBC>(object), _preset(object._preset)
  {
    _solution_tag = object._solution_tag;
  }

  virtual bool preset() const override { return _preset; }

  virtual void presetSolution(TagID tag) override
  {
    _solution_tag = tag;

    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::IndexType<size_t>>(0, this->numBoundaryNodes()),
        *static_cast<DirichletBC *>(this));
  }

  using GPUNodalBC<DirichletBC>::operator();
  using GPUNodalBC<DirichletBC>::_gpu_var;

  KOKKOS_FUNCTION void operator()(const size_t tid) const
  {
    auto bc = static_cast<const DirichletBC *>(this);
    auto node = this->boundaryNodeID(tid);

    ResidualNodalDatum datum(node, this->systems(), _gpu_var);

    auto & system = this->systems()[_gpu_var.sys()];
    auto dof = system.getNodeLocalDofIndex(node, _gpu_var.var());

    system.getVectorDofValue(dof, _solution_tag) = bc->computeValue(datum);
  }

  KOKKOS_FUNCTION Real computeQpResidual(const ResidualNodalDatum & datum) const
  {
    auto bc = static_cast<const DirichletBC *>(this);

    return datum.u() - bc->computeValue(datum);
  }

private:
  // Whether or not the value is to be preset
  const bool _preset;
  // The solution tag to be preset
  TagID _solution_tag;
};
