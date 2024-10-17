//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#ifndef MOOSE_IGNORE_LIBCEED

#include "CEEDAssembly.h"

#include "MooseMesh.h"
#include "MooseVariableFieldBase.h"
#include "FEProblemBase.h"
#include "NonlinearSystemBase.h"
#include "AuxiliarySystem.h"
#include "Assembly.h"

void
CEEDAssembly::init(FEProblemBase & problem)
{
  _problem = &problem;
  _ceed = _problem->getMooseApp().getCeed();
  _mesh = &_problem->mesh();
  _assembly = &_problem->assembly(0, 0);

  _q_order.clear();
  _variables.clear();
  _elem_types.clear();
  _quadratures.clear();

  // Set the list of unique FE types, element types, and quadrature orders

  auto getVariables = [&](SystemBase & system)
  {
    for (auto var : system.getVariables(0))
      _variables.push_back(var);
  };

  for (unsigned int nl = 0; nl < _problem->numNonlinearSystems(); ++nl)
    getVariables(_problem->getNonlinearSystemBase(nl));

  getVariables(_problem->getAuxiliarySystem());

  for (auto elem : *_mesh->getActiveLocalElementRange())
  {
    if (!_elem_types.count(elem->type()))
      _elem_types[elem->type()] = elem;

    _assembly->setCurrentSubdomainID(elem->subdomain_id());
    _assembly->setVolumeQRule(elem);

    auto qrule = _assembly->writeableQRule();

    _q_order[elem] = qrule->get_order();

    if (!_quadratures.count(qrule->get_order()))
      _quadratures[qrule->get_order()] = qrule;
  }

  // Generate CEED basis

  for (auto var : _variables)
    for (auto [elem_type, elem] : _elem_types)
      for (auto [order, qrule] : _quadratures)
      {
        auto tuple = std::make_tuple(var, elem_type, order);
        auto & basis = _basis[tuple];
        auto fe = FEBase::build(elem->dim(), var->feType());

        // Initialize quadrature rule with element type to get reference quadrature points

        qrule->init(elem_type);

        auto q_points = qrule->get_points();
        auto q_weight = qrule->get_weights();

        CeedInt num_qpts = qrule->n_points();

        std::vector<CeedScalar> q_ref(elem->dim() * num_qpts);

        for (CeedInt qp = 0; qp < num_qpts; ++qp)
          for (unsigned int d = 0; d < elem->dim(); ++d)
            q_ref[d + elem->dim() * qp] = q_points[qp](d);

        // Initialize FE with the representative element pointer

        fe->attach_quadrature_rule(qrule);
        fe->reinit(elem);

        auto & phi = fe->get_phi();
        auto & dphidxi = fe->get_dphidxi();
        auto & dphideta = fe->get_dphideta();
        auto & dphidzeta = fe->get_dphidzeta();

        CeedInt num_nodes = phi.size();

        std::vector<CeedScalar> interp(num_qpts * num_nodes);
        std::vector<CeedScalar> grad(elem->dim() * num_qpts * num_nodes);

        for (CeedInt i = 0; i < num_nodes; ++i)
          for (CeedInt qp = 0; qp < num_qpts; ++qp)
          {
            interp[qp + num_qpts * i] = phi[i][qp];

            if (elem->dim() >= 1)
              grad[0 + elem->dim() * (qp + num_qpts * i)] = dphidxi[i][qp];
            if (elem->dim() >= 2)
              grad[1 + elem->dim() * (qp + num_qpts * i)] = dphideta[i][qp];
            if (elem->dim() == 3)
              grad[2 + elem->dim() * (qp + num_qpts * i)] = dphidzeta[i][qp];
          }

        // Topology is only used for getting dimensions

        CeedElemTopology topo;

        if (elem->dim() == 1)
          topo = CEED_TOPOLOGY_LINE;
        else if (elem->dim() == 2)
          topo = CEED_TOPOLOGY_QUAD;
        else if (elem->dim() == 3)
          topo = CEED_TOPOLOGY_HEX;

        CeedBasisCreateH1(_ceed,
                          topo,
                          var->count(),
                          num_nodes,
                          num_qpts,
                          interp.data(),
                          grad.data(),
                          q_ref.data(),
                          q_weight.data(),
                          &basis);
      }
}

#endif
