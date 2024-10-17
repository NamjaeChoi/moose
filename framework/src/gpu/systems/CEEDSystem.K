//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#ifndef MOOSE_IGNORE_LIBCEED

#include "CEEDSystem.h"

#include "MooseMesh.h"
#include "Assembly.h"
#include "NonlinearSystemBase.h"

#include "libmesh/system.h"

void
CEEDSystem::setupDofs()
{
  auto & sys = _system->system();
  auto & dof_map = _system->dofMap();
  auto solution = dynamic_cast<PetscVector<Number> *>(sys.current_local_solution.get());

  CeedInt dim = _mesh->dimension();

  _n_local_dofs = sys.n_local_dofs();
  _n_ghost_dofs = dof_map.get_send_list().size();

  // Create CEED restrictions

  _rstr.createElementRestriction(*_ceed_assembly, _system->getVariables(0));
  _rstr.createQuadratureRestriction(*_ceed_assembly, _system->getVariables(0));

  // Setup DOF communication maps

  _comm = &dof_map.comm();

  std::vector<std::vector<dof_id_type>> send_list(_comm->size());
  std::vector<std::vector<dof_id_type>> recv_list(_comm->size());

  for (auto dof : dof_map.get_send_list())
    recv_list[dof_map.dof_owner(dof)].push_back(dof);

  for (processor_id_type proc = 0; proc < _comm->size(); proc++)
    _comm->scatter(recv_list, send_list[proc], proc);

  _local_comm_list.create(_comm->size());
  _ghost_comm_list.create(_comm->size());

  for (processor_id_type proc = 0; proc < _comm->size(); proc++)
  {
    _local_comm_list[proc].create(send_list[proc].size());
    _ghost_comm_list[proc].create(recv_list[proc].size());

    for (dof_id_type i = 0; i < send_list[proc].size(); ++i)
      _local_comm_list[proc][i] = solution->map_global_to_local_index(send_list[proc][i]);

    for (dof_id_type i = 0; i < recv_list[proc].size(); ++i)
      _ghost_comm_list[proc][i] = solution->map_global_to_local_index(recv_list[proc][i]);
  }

  _local_comm_list.copyNested();
  _ghost_comm_list.copyNested();
}

void
CEEDSystem::setupOperators()
{
  _qf_elem_interp.clear();
  _qf_elem_grad.clear();
  _op_elem_interp.clear();
  _op_elem_grad.clear();

  for (auto var : _system->getVariables(0))
    for (auto type : _ceed_assembly->getUniqueElemTypes())
      for (auto qrule : _ceed_assembly->getUniqueQuadratures())
      {
        auto tuple = std::make_tuple(var, type, qrule->get_order());
        auto basis = _ceed_assembly->getBasis(tuple);
        auto elem_rstr = _rstr.getElementRestriction(tuple);
        auto qp_rstr_interp = _rstr.getQuadratureRestriction(tuple, CEED_EVAL_INTERP);
        auto qp_rstr_grad = _rstr.getQuadratureRestriction(tuple, CEED_EVAL_GRAD);
        auto & qf_interp = _qf_elem_interp[tuple];
        auto & qf_grad = _qf_elem_grad[tuple];
        auto & op_interp = _op_elem_interp[tuple];
        auto & op_grad = _op_elem_grad[tuple];

        CeedQFunctionCreateIdentity(
            _ceed, var->count(), CEED_EVAL_INTERP, CEED_EVAL_NONE, &qf_interp);
        CeedQFunctionCreateIdentity(
            _ceed, var->count() * _mesh->dimension(), CEED_EVAL_GRAD, CEED_EVAL_NONE, &qf_grad);
        CeedOperatorCreate(_ceed, qf_interp, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_interp);
        CeedOperatorCreate(_ceed, qf_grad, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_grad);
        CeedOperatorSetField(op_interp, "input", elem_rstr, basis, CEED_VECTOR_ACTIVE);
        CeedOperatorSetField(
            op_interp, "output", qp_rstr_interp, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);
        CeedOperatorSetField(op_grad, "input", elem_rstr, basis, CEED_VECTOR_ACTIVE);
        CeedOperatorSetField(op_grad, "output", qp_rstr_grad, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);
      }
}

void
CEEDSystem::init(SystemBase & system, CEEDAssembly & ceed_assembly)
{
  _ceed_assembly = &ceed_assembly;
  _system = &system;
  _mesh = &_system->mesh();
  _ceed = _system->subproblem().getMooseApp().getCeed();

  setupDofs();
  setupOperators();

  _initialized = true;
}

void
CEEDSystem::sync(GPUMemcpyKind dir)
{
  if (dir == GPUMemcpyKind::HOST_TO_DEVICE)
  {
    for (auto tag : _active_variable_tags)
    {
      auto & vector = _system->getVector(tag);

      _vectors[tag].create(vector, *this);
      _vectors[tag].copy(dir);
    }

    for (auto tag : _active_residual_tags)
    {
      auto & vector = _system->getVector(tag);

      _vectors[tag].create(vector, *this, true);
      _vectors[tag] = 0;
    }

    for (auto tag : _active_matrix_tags)
    {
      auto & matrix = _system->getMatrix(tag);

      _matrices[tag].create(matrix);
      _matrices[tag] = 0;
    }
  }
  else if (dir == GPUMemcpyKind::DEVICE_TO_HOST)
  {
    for (auto tag : _active_variable_tags)
      _vectors[tag].restore();

    for (auto tag : _active_residual_tags)
      _vectors[tag].close();

    for (auto tag : _active_matrix_tags)
      _matrices[tag].close();
  }
}

void
CEEDSystem::sync(std::set<TagID> tags, GPUMemcpyKind dir)
{
  if (dir == GPUMemcpyKind::HOST_TO_DEVICE)
  {
    for (auto tag : tags)
    {
      if (!_system->hasVector(tag))
        continue;

      auto & vector = _system->getVector(tag);

      _vectors[tag].create(vector, *this);
      _vectors[tag].copy(dir);
    }
  }
  else if (dir == GPUMemcpyKind::DEVICE_TO_HOST)
  {
    for (auto tag : tags)
    {
      if (!_system->hasVector(tag))
        continue;

      _vectors[tag].copy(dir);
      _vectors[tag].restore();
    }
  }
}

void
CEEDSystem::setActiveVariables(std::set<MooseVariableFieldBase *> vars)
{
  _active_variables.clear();

  for (auto var : vars)
    if (var->sys().number() == _system->number())
      _active_variables.insert(var);
}

void
CEEDSystem::setActiveVariableTags(std::set<TagID> tags)
{
  if (!_active_variables.size())
    return;

  _active_variable_tags.clear();

  for (auto tag : tags)
    if (_system->hasVector(tag))
      _active_variable_tags.insert(tag);
}

void
CEEDSystem::setActiveResidualTags(std::set<TagID> tags)
{
  _active_residual_tags.clear();

  for (auto tag : tags)
    if (_system->hasVector(tag))
      _active_residual_tags.insert(tag);
}

void
CEEDSystem::setActiveMatrixTags(std::set<TagID> tags)
{
  _active_matrix_tags.clear();

  for (auto tag : tags)
    if (_system->hasMatrix(tag))
      _active_matrix_tags.insert(tag);
}

void
CEEDSystem::project()
{
  std::vector<CeedRequest> requests;

  for (auto tag : _active_variable_tags)
    for (auto var : _active_variables)
      for (auto type : _ceed_assembly->getUniqueElemTypes())
        for (auto qrule : _ceed_assembly->getUniqueQuadratures())
        {
          auto tuple = std::make_tuple(var, type, qrule->get_order());
          auto qp_rstr_interp = _rstr.getQuadratureRestriction(tuple, CEED_EVAL_INTERP);
          auto qp_rstr_grad = _rstr.getQuadratureRestriction(tuple, CEED_EVAL_GRAD);
          auto & q_interp = _q_elem_interp[tag][tuple];
          auto & q_grad = _q_elem_grad[tag][tuple];
          auto & op_interp = _op_elem_interp[tuple];
          auto & op_grad = _op_elem_grad[tuple];

          CeedOperatorApply(op_interp, _vectors[tag], q_interp, &requests.emplace_back());
          CeedOperatorApply(op_grad, _vectors[tag], q_grad, &requests.emplace_back());
        }

  for (auto request : requests)
    CeedRequestWait(&request);
}

#endif
