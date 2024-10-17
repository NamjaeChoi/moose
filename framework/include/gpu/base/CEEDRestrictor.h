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

#include "CEEDAssembly.h"

#include "MooseMesh.h"

class MooseVariableFieldBase;

class CEEDRestrictor
{
public:
  /**
   * Create the CEED element restriction
   * @param assembly The CEED assembly
   * @param variables The MOOSE field variables to restrict
   * @param blocks The blocks to restrict
   * @param boundaries The boundaries to restrict
   */
  void createElementRestriction(const CEEDAssembly & assembly,
                                const std::vector<MooseVariableFieldBase *> & variables,
                                const std::set<SubdomainID> * blocks = nullptr,
                                const std::set<BoundaryID> * boundaries = nullptr);
  void createElementRestriction(const CEEDAssembly & assembly,
                                const std::set<MooseVariableFieldBase *> & variables,
                                const std::set<SubdomainID> * blocks = nullptr,
                                const std::set<BoundaryID> * boundaries = nullptr)
  {
    createElementRestriction(
        assembly,
        std::vector<MooseVariableFieldBase *>(variables.begin(), variables.end()),
        blocks,
        boundaries);
  }

  /**
   * Create the CEED quadrature restriction
   * @param assembly The CEED assembly
   * @param variables The MOOSE field variables to restrict
   * @param blocks The blocks to restrict
   * @param boundaries The boundaries to restrict
   */
  void createQuadratureRestriction(const CEEDAssembly & assembly,
                                   const std::vector<MooseVariableFieldBase *> & variables,
                                   const std::set<SubdomainID> * blocks = nullptr,
                                   const std::set<BoundaryID> * boundaries = nullptr);
  void createQuadratureRestriction(const CEEDAssembly & assembly,
                                   const std::set<MooseVariableFieldBase *> & variables,
                                   const std::set<SubdomainID> * blocks = nullptr,
                                   const std::set<BoundaryID> * boundaries = nullptr)
  {
    createQuadratureRestriction(
        assembly,
        std::vector<MooseVariableFieldBase *>(variables.begin(), variables.end()),
        blocks,
        boundaries);
  }

  /**
   * Get the CEED element restriction
   * @param tuple The tuple consisting of MOOSE field variable, element type, quadrature rule
   */
  auto getElementRestriction(ceed_tuple tuple) const { return _elem_rstr.at(tuple); }

  /**
   * Get the CEED quadrature restriction
   * @param tuple The tuple consisting of MOOSE field variable, element type, quadrature rule
   * @param mode CEED evaluation mode
   */
  auto getQuadratureRestriction(ceed_tuple tuple, CeedEvalMode mode = CEED_EVAL_INTERP) const
  {
    switch (mode)
    {
      case CEED_EVAL_INTERP:
        return _qp_rstr_interp.at(tuple);
      case CEED_EVAL_GRAD:
        return _qp_rstr_grad.at(tuple);
      default:
        return CEEDElemRestriction();
    }
  }

private:
  // CEED element restrictions
  std::map<ceed_tuple, CEEDElemRestriction> _elem_rstr;
  // CEED quadrature restrictions for interpolation
  std::map<ceed_tuple, CEEDElemRestriction> _qp_rstr_interp;
  // CEED quadrature restrictions for gradient
  std::map<ceed_tuple, CEEDElemRestriction> _qp_rstr_grad;
};

#endif
