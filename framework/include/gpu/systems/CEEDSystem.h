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

#include "GPUVector.h"
#include "GPUMatrix.h"
#include "CEEDAssembly.h"
#include "CEEDRestrictor.h"

#include "SystemBase.h"
#include "MooseVariableFieldBase.h"

#include "libmesh/communicator.h"

class MooseMesh;
class SystemBase;

class CEEDSystem
{
private:
  void setupDofs();
  void setupOperators();

public:
  /**
   * Initialize the CEED system
   * @param system Associated MOOSE system
   * @param ceed_assembly CEED assembly
   */
  void init(SystemBase & system, CEEDAssembly & ceed_assembly);
  /**
   * Check whether the CEED system was initialized
   */
  auto initialized() { return _initialized; }
  /**
   * Prepare and copy in or out vectors and matrices
   * @param dir Copy direction
   */
  void sync(GPUMemcpyKind dir);
  /**
   * Copy in or out vectors corresponding to the given tags
   * @param tags A set of vector tags to be copied in or out
   * @param dir Copy direction
   */
  void sync(std::set<TagID> tags, GPUMemcpyKind dir);
  /**
   * Compute quadrature point values and gradients of active variables
   */
  void project();
  /**
   * Cache active variables
   * @param vars A set of MOOSE variables to be cached
   */
  void setActiveVariables(std::set<MooseVariableFieldBase *> vars);
  /**
   * Cache active variable tags
   * @param vars A set of variable tags to be cached
   */
  void setActiveVariableTags(std::set<TagID> tags);
  /**
   * Cache active residual tags
   * @param vars A set of residual tags to be cached
   */
  void setActiveResidualTags(std::set<TagID> tags);
  /**
   * Cache active matrix tags
   * @param vars A set of matrix tags to be cached
   */
  void setActiveMatrixTags(std::set<TagID> tags);
  /**
   * Get active variables
   */
  auto getActiveVariables() { return _active_variables; }
  /**
   * Get active variable tags
   */
  auto getActiveVariableTags() { return _active_variable_tags; }
  /**
   * Get active residual tags
   */
  auto getActiveResidualTags() { return _active_residual_tags; }
  /**
   * Get active matrix tags
   */
  auto getActiveMatrixTags() { return _active_matrix_tags; }
  /**
   * Clear cached active variables
   */
  void clearActiveVariables() { _active_variables.clear(); }
  /**
   * Clear cached active variable tags
   */
  void clearActiveVariableTags() { _active_variable_tags.clear(); }
  /**
   * Clear cached active residual tags
   */
  void clearActiveResidualTags() { _active_residual_tags.clear(); }
  /**
   * Clear cached active matrix tags
   */
  void clearActiveMatrixTags() { _active_matrix_tags.clear(); }
  /**
   * Get the CEED logical device
   */
  auto getCeed() const { return _ceed; }
  /**
   * Get the MOOSE system
   */
  const auto & getSystem() const { return *_system; }
  /**
   * Get the libMesh DOF map
   */
  const auto & getDofMap() const { return _system->dofMap(); }
  /**
   * Get the libMesh communicator
   */
  const auto & getComm() const { return *_comm; }
  /**
   * Get the list of local DOF indices to send/receive
   */
  const auto & getLocalCommList() const { return _local_comm_list; }
  /**
   * Get the list of ghost DOF indices to send/receive
   */
  const auto & getGhostCommList() const { return _ghost_comm_list; }
  /**
   * Get the number of local DOFs
   */
  auto getNumLocalDofs() const { return _n_local_dofs; }
  /**
   * Get the number of ghost DOFs
   */
  auto getNumGhostDofs() const { return _n_ghost_dofs; }
  /**
   * Get the vector associated with a tag
   * @param tag Vector tag
   */
  auto & getVector(TagID tag) const { return _vectors.at(tag); }
  /**
   * Get the quadrature vector
   * @param tuple The tuple consisting of MOOSE field variable, element type, quadrature rule
   * @param tag Vector tag
   * @param mode CEED evaluation mode
   */
  auto getQuadratureVector(ceed_tuple tuple, TagID tag, CeedEvalMode mode)
  {
    CEEDVector vec;
    CEEDElemRestriction rstr;

    switch (mode)
    {
      case CEED_EVAL_INTERP:
        vec = _q_elem_interp[tag][tuple];
        rstr = _rstr.getQuadratureRestriction(tuple, CEED_EVAL_INTERP);
        break;
      case CEED_EVAL_GRAD:
        vec = _q_elem_grad[tag][tuple];
        rstr = _rstr.getQuadratureRestriction(tuple, CEED_EVAL_GRAD);
        break;
      default:
        return vec;
    }

    if (!vec)
    {
      CeedElemRestrictionCreateVector(rstr, NULL, &vec);
      CeedVectorSetValue(vec, 0);
    }

    return vec;
  }

private:
  // Whether the GPU system was initialized
  bool _initialized = false;
  // CEED logical device
  Ceed _ceed;
  // CEED vectors
  std::map<TagID, GPUVector> _vectors;
  // CEED matrices
  std::map<TagID, GPUMatrix> _matrices;

private:
  // Pointer to the CEED assembly
  CEEDAssembly * _ceed_assembly;
  // Pointer to the MOOSE system
  SystemBase * _system = nullptr;
  // Pointer to the MOOSE mesh
  const MooseMesh * _mesh = nullptr;
  // Pointer to the libMesh communicator
  const Parallel::Communicator * _comm = nullptr;
  // Number of local DOFs
  dof_id_type _n_local_dofs = 0;
  // Number of ghost DOFs
  dof_id_type _n_ghost_dofs = 0;

private:
  // CEED element quadrature value vectors
  std::map<TagID, std::map<ceed_tuple, CEEDVector>> _q_elem_interp;
  // CEED element quadrature gradient vectors
  std::map<TagID, std::map<ceed_tuple, CEEDVector>> _q_elem_grad;
  // CEED element interpolation QFunctions
  std::map<ceed_tuple, CEEDQFunction> _qf_elem_interp;
  // CEED element gradient QFunctions
  std::map<ceed_tuple, CEEDQFunction> _qf_elem_grad;
  // CEED element interpolation operators
  std::map<ceed_tuple, CEEDOperator> _op_elem_interp;
  // CEED element gradient operators
  std::map<ceed_tuple, CEEDOperator> _op_elem_grad;
  // CEED restrictor
  CEEDRestrictor _rstr;
  // List of active variable numbers
  std::set<MooseVariableFieldBase *> _active_variables;
  // List of active variable tags
  std::set<TagID> _active_variable_tags;
  // List of active residual tags
  std::set<TagID> _active_residual_tags;
  // List of active matrix tags
  std::set<TagID> _active_matrix_tags;
  // List of DOFs to send and receive
  GPUArray<GPUArray<dof_id_type>> _local_comm_list;
  GPUArray<GPUArray<dof_id_type>> _ghost_comm_list;
};

#endif
