//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#ifndef MOOSE_IGNORE_LIBCEED

#pragma once

#include "CEEDHeader.h"
#include "GPUArray.h"

#include "libmesh/elem.h"
#include "libmesh/fe_base.h"
#include "libmesh/fe_type.h"
#include "libmesh/quadrature.h"

class FEProblemBase;
class MooseMesh;
class MooseVariableFieldBase;
class Assembly;

using ceed_tuple = std::tuple<MooseVariableFieldBase *, ElemType, unsigned int>;

class CEEDAssembly
{
public:
  // Initialize assembly
  void init(FEProblemBase & problem);

  auto getMesh() const { return _mesh; }

  auto getCeed() const { return _ceed; }

  auto getBasis(ceed_tuple tuple) const
  {
    if (!_basis.count(tuple))
      mooseError("CEEDAssembly error: requested basis was not set.");

    return _basis.at(tuple);
  }

  auto getUniqueElemTypes() const
  {
    std::set<ElemType> types;

    for (auto [type, elem] : _elem_types)
      types.insert(type);

    return types;
  }

  auto getUniqueQuadratures() const
  {
    std::set<QBase *> qrules;

    for (auto [order, qrule] : _quadratures)
      qrules.insert(qrule);

    return qrules;
  }

  auto getQuadratureOrder(const Elem * elem) const { return _q_order.at(elem); }

private:
  // CEED logical device
  Ceed _ceed;
  // Pointer to MOOSE problem
  FEProblemBase * _problem = nullptr;
  // Pointer to MOOSE mesh
  MooseMesh * _mesh = nullptr;
  // Pointer to MOOSE assembly
  Assembly * _assembly = nullptr;

private:
  // Quadrature order of each element
  std::map<const Elem *, unsigned int> _q_order;
  // All MOOSE variables
  std::vector<MooseVariableFieldBase *> _variables;
  // Unique element types and representative elements
  std::map<ElemType, const Elem *> _elem_types;
  // Unique quadrature orders and associated QBase
  std::map<unsigned int, QBase *> _quadratures;
  // CEED basis
  std::map<ceed_tuple, CEEDBasis> _basis;
};

#endif
