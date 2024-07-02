//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "GPUMaterialBase.h"
#include "GPUDatum.h"

#include "Coupleable.h"
#include "MaterialPropertyInterface.h"

template <typename Material>
class GPUMaterial : public GPUMaterialBase, public Coupleable, public MaterialPropertyInterface
{
public:
  static InputParameters validParams()
  {
    InputParameters params = GPUMaterialBase::validParams();
    params += MaterialPropertyInterface::validParams();
    params.addParamNamesToGroup("use_displaced_mesh", "Advanced");
    return params;
  }

  // Constructor
  GPUMaterial(const InputParameters & parameters)
    : GPUMaterialBase(parameters),
      Coupleable(this, false),
      MaterialPropertyInterface(this, blockIDs(), boundaryIDs()),
      _bnd(_material_data_type != Moose::BLOCK_MATERIAL_DATA),
      _neighbor(_material_data_type == Moose::NEIGHBOR_MATERIAL_DATA),
      _qrule(_bnd ? (_neighbor ? _subproblem.assembly(_tid, 0).qRuleNeighbor()
                               : _subproblem.assembly(_tid, 0).qRuleFace())
                  : _subproblem.assembly(_tid, 0).qRule())
  {
    for (auto coupled_var : getCoupledMooseVars())
      addMooseVariableDependency(coupled_var);
  }

  // Copy constructor
  GPUMaterial(const GPUMaterial & object)
    : GPUMaterialBase(object),
      Coupleable(&object, false, false, false),
      MaterialPropertyInterface(&object, object.blockIDs(), object.boundaryIDs(), false),
      _bnd(object._bnd),
      _neighbor(object._neighbor),
      _qrule(object._qrule)
  {
  }

  virtual bool isBoundaryMaterial() const override { return _bnd; }

  virtual const std::unordered_set<unsigned int> & getMatPropDependencies() const override
  {
    return MaterialPropertyInterface::getMatPropDependencies();
  }

  // Dispatch material property calculation to GPU
  virtual void computeProperties() override
  {
    if (getMatPropDependencies().size())
      Kokkos::fence();

    if (!_bnd && !_neighbor)
      Kokkos::parallel_for(
          Kokkos::RangePolicy<ElementLoop, Kokkos::IndexType<size_t>>(0, numBlockElements()),
          *static_cast<Material *>(this));
    else if (_bnd && !_neighbor)
      Kokkos::parallel_for(
          Kokkos::RangePolicy<SideLoop, Kokkos::IndexType<size_t>>(0, numBlockSides()),
          *static_cast<Material *>(this));
    else
      Kokkos::parallel_for(
          Kokkos::RangePolicy<NeighborLoop, Kokkos::IndexType<size_t>>(0, numBlockSides()),
          *static_cast<Material *>(this));
  }

  // Overloaded operators called by Kokkos::parallel_for
  KOKKOS_FUNCTION void operator()(ElementLoop, const size_t tid) const
  {
    auto material = static_cast<const Material *>(this);
    auto elem = blockElementID(tid);

    Datum datum(elem, assembly(), systems());

    for (unsigned int qp = 0; qp < datum.n_qps(); qp++)
      material->computeQpProperties(datum, qp);
  }
  KOKKOS_FUNCTION void operator()(SideLoop, const size_t tid) const
  {
    auto material = static_cast<const Material *>(this);
    auto elem = blockElementSideID(tid);

    Datum datum(elem.first, elem.second, assembly(), systems());

    for (unsigned int qp = 0; qp < datum.n_qps(); qp++)
      material->computeQpProperties(datum, qp);
  }
  KOKKOS_FUNCTION void operator()(NeighborLoop, const size_t tid) const
  {
    auto material = static_cast<const Material *>(this);
    auto elem = blockElementSideID(tid);

    NeighborDatum datum(elem.first, elem.second, assembly(), systems());

    if (datum.hasNeighbor())
      for (unsigned int qp = 0; qp < datum.n_qps(); qp++)
        material->computeQpProperties(datum, qp);
  }

protected:
  const bool _bnd;
  const bool _neighbor;

  virtual const MaterialData & materialData() const override { return _material_data; }
  virtual MaterialData & materialData() override { return _material_data; }

  // Dummy members that should not be accessed by derived classes
private:
  const QBase * const & _qrule;

  virtual const QBase & qRule() const override { return *_qrule; }
};
