//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "GPUSystem.h"
#include "GPUNodalBCBase.h"

#include "MooseMesh.h"
#include "Assembly.h"
#include "NonlinearSystemBase.h"

#include "libmesh/system.h"

void
GPUSystem::setupVariables()
{
  auto & sys = _system->system();

  _n_vars = sys.n_vars();

  _var_fe_types.create(_n_vars);
  _var_subdomain_active.create(_n_vars, _mesh->meshSubdomains().size());

  for (unsigned int var = 0; var < _n_vars; ++var)
  {
    _var_fe_types[var] = _gpu_assembly.getFETypeNum(sys.variable_type(var));

    for (auto subdomain : _mesh->meshSubdomains())
      _var_subdomain_active(var, _mesh->getGPUSubdomainID(subdomain)) =
          sys.variable(var).active_on_subdomain(subdomain);
  }

  _var_fe_types.copy();
  _var_subdomain_active.copy();

  // Set coupling

  auto nl_system = dynamic_cast<NonlinearSystemBase *>(_system);

  if (nl_system)
  {
    _coupling.create(_n_vars);

    std::map<unsigned int, std::vector<unsigned int>> coupling;

    auto & ce = _system->feProblem().couplingEntries(0, nl_system->number());

    for (const auto & [ivar, jvar] : ce)
      coupling[ivar->number()].push_back(jvar->number());

    for (unsigned int var = 0; var < _n_vars; ++var)
      _coupling[var] = coupling[var];

    _coupling.copy();
  }
}

void
GPUSystem::setupDofs()
{
  auto & sys = _system->system();
  auto & dof_map = _system->dofMap();

  _n_elems = _mesh->getGPULocalElementMap().size();
  _n_nodes = _mesh->getGPULocalNodeMap().size();
  _n_local_dofs = sys.n_local_dofs();
  _n_ghost_dofs = dof_map.get_send_list().size();

  _residual_tag_active.create(MAX_TAG);
  _residual_tag_active = false;

  _matrix_tag_active.create(MAX_TAG);
  _matrix_tag_active = false;

  _vectors.create(MAX_TAG);
  _matrices.create(MAX_TAG);

  _local_elem_dof_index.create(_n_vars);
  _local_node_dof_index.create(_n_vars);
  _local_to_global_dof_index.create(_n_local_dofs + _n_ghost_dofs);

  _max_dofs_per_elem.create(_n_vars);
  _max_dofs_per_elem = 0;

  auto solution = dynamic_cast<PetscVector<Number> *>(sys.current_local_solution.get());

  std::vector<dof_id_type> dof_indices;

  for (unsigned int var = 0; var < _n_vars; ++var)
  {
    for (auto & [elem, id] : _mesh->getGPULocalElementMap())
    {
      dof_map.dof_indices(elem, dof_indices, var);

      _max_dofs_per_elem[var] = std::max(_max_dofs_per_elem[var], dof_indices.size());
    }

    _local_elem_dof_index[var].create(_n_elems, _max_dofs_per_elem[var]);
    _local_elem_dof_index[var] = -1;

    _local_node_dof_index[var].create(_n_nodes);
    _local_node_dof_index[var] = -1;

    for (auto & [elem, id] : _mesh->getGPULocalElementMap())
    {
      dof_map.dof_indices(elem, dof_indices, var);

      for (unsigned int dof = 0; dof < dof_indices.size(); ++dof)
      {
        _local_elem_dof_index[var](id, dof) = solution->map_global_to_local_index(dof_indices[dof]);
        _local_to_global_dof_index[_local_elem_dof_index[var](id, dof)] = dof_indices[dof];
      }
    }

    for (auto & [node, id] : _mesh->getGPULocalNodeMap())
    {
      dof_map.dof_indices(node, dof_indices, var);

      if (dof_indices.size())
        _local_node_dof_index[var][id] = solution->map_global_to_local_index(dof_indices[0]);
    }
  }

  _local_elem_dof_index.copyNested();
  _local_node_dof_index.copyNested();
  _local_to_global_dof_index.copy();

  _max_dofs_per_elem.copy();

  // Setup DOF communication maps

  _comm = &dof_map.comm();
  _n_procs = _comm->size();

  std::vector<std::vector<dof_id_type>> send_list(_n_procs);
  std::vector<std::vector<dof_id_type>> recv_list(_n_procs);

  for (auto dof : dof_map.get_send_list())
    recv_list[dof_map.dof_owner(dof)].push_back(dof);

  for (processor_id_type proc = 0; proc < _n_procs; proc++)
    _comm->scatter(recv_list, send_list[proc], proc);

  _local_comm_list.create(_n_procs);
  _ghost_comm_list.create(_n_procs);

  for (processor_id_type proc = 0; proc < _n_procs; proc++)
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
GPUSystem::setupSparsity()
{
  auto & dof_map = _system->dofMap();
  auto pattern = dof_map.get_sparsity_pattern();

  std::vector<PetscInt> col_idx;
  std::vector<PetscInt> row_idx;
  std::vector<PetscInt> row_ptr = {0};

  for (PetscInt r = dof_map.first_dof(); r < dof_map.end_dof(); ++r)
  {
    auto & cols = pattern->get_sparsity_pattern().at(r - dof_map.first_dof());

    for (auto col : cols)
    {
      col_idx.push_back(col);
      row_idx.push_back(r);
    }

    row_ptr.push_back(col_idx.size());
  }

  std::vector<std::vector<PetscInt>> send_cols(_n_procs), send_num_cols(_n_procs);
  std::vector<std::vector<PetscInt>> recv_cols(_n_procs), recv_num_cols(_n_procs);

  for (processor_id_type proc = 0; proc < _n_procs; proc++)
    for (auto r : _local_comm_list[proc])
    {
      for (PetscInt c = row_ptr[r]; c < row_ptr[r + 1]; ++c)
        send_cols[proc].push_back(col_idx[c]);

      send_num_cols[proc].push_back(row_ptr[r + 1] - row_ptr[r]);
    }

  for (processor_id_type proc = 0; proc < _n_procs; proc++)
  {
    _comm->scatter(send_cols, recv_cols[proc], proc);
    _comm->scatter(send_num_cols, recv_num_cols[proc], proc);
  }

  std::vector<PetscInt> col(_n_procs), row(_n_procs);

  for (auto r : dof_map.get_send_list())
  {
    auto proc = dof_map.dof_owner(r);
    auto n_cols = recv_num_cols[proc][row[proc]++];

    for (PetscInt c = 0; c < n_cols; ++c)
    {
      col_idx.push_back(recv_cols[proc][col[proc]++]);
      row_idx.push_back(r);
    }

    row_ptr.push_back(col_idx.size());
  }

  _sparsity.col_idx = col_idx;
  _sparsity.row_idx = row_idx;
  _sparsity.row_ptr = row_ptr;
}

void
GPUSystem::checkNodalBCs()
{
  auto nl_system = static_cast<NonlinearSystemBase *>(_system);

  _nbc_residual_dof.create(MAX_TAG);
  _nbc_matrix_dof.create(MAX_TAG);

  for (auto vector_tag : _system->feProblem().getVectorTags())
  {
    auto tag = vector_tag._id;

    for (auto bc :
         nl_system->getGPUNodalBCWarehouse().getVectorTagObjectWarehouse(tag, 0).getObjects())
    {
      auto nbc = static_cast<GPUNodalBCBase *>(bc.get());
      auto var = nbc->variable().number();

      if (!_nbc_residual_dof[tag].isAlloc())
      {
        _nbc_residual_dof[tag].create(_n_local_dofs + _n_ghost_dofs);
        _nbc_residual_dof[tag] = false;
      }

      for (auto node : nbc->getNodes())
        _nbc_residual_dof[tag][_local_node_dof_index[var][node]] = true;

      // Let remote processes know about ghost DOFs associated with nodal BCs not to compute
      // residual on them

      std::vector<std::vector<char>> send(_n_procs), recv(_n_procs);

      for (processor_id_type proc = 0; proc < _n_procs; proc++)
        for (auto dof : _local_comm_list[proc])
          send[proc].push_back(_nbc_residual_dof[tag][dof]);

      for (processor_id_type proc = 0; proc < _n_procs; proc++)
        _comm->scatter(send, recv[proc], proc);

      for (processor_id_type proc = 0; proc < _n_procs; proc++)
        for (dof_id_type i = 0; i < _ghost_comm_list[proc].size(); ++i)
          _nbc_residual_dof[tag][_ghost_comm_list[proc][i]] = recv[proc][i];
    }
  }

  for (auto matrix_tag : _system->feProblem().getMatrixTags())
  {
    auto tag = matrix_tag.second;

    for (auto bc :
         nl_system->getGPUNodalBCWarehouse().getMatrixTagObjectWarehouse(tag, 0).getObjects())
    {
      auto nbc = static_cast<GPUNodalBCBase *>(bc.get());
      auto var = nbc->variable().number();

      if (!_nbc_matrix_dof[tag].isAlloc())
      {
        _nbc_matrix_dof[tag].create(_n_local_dofs + _n_ghost_dofs);
        _nbc_matrix_dof[tag] = false;
      }

      for (auto node : nbc->getNodes())
        _nbc_matrix_dof[tag][_local_node_dof_index[var][node]] = true;

      // Let remote processes know about ghost DOFs associated with nodal BCs not to compute
      // Jacobian on them

      std::vector<std::vector<char>> send(_n_procs), recv(_n_procs);

      for (processor_id_type proc = 0; proc < _n_procs; proc++)
        for (auto dof : _local_comm_list[proc])
          send[proc].push_back(_nbc_matrix_dof[tag][dof]);

      for (processor_id_type proc = 0; proc < _n_procs; proc++)
        _comm->scatter(send, recv[proc], proc);

      for (processor_id_type proc = 0; proc < _n_procs; proc++)
        for (dof_id_type i = 0; i < _ghost_comm_list[proc].size(); ++i)
          _nbc_matrix_dof[tag][_ghost_comm_list[proc][i]] = recv[proc][i];
    }
  }

  _nbc_residual_dof.copyNested();
  _nbc_matrix_dof.copyNested();
}

void
GPUSystem::init(SystemBase & system, const GPUAssembly & gpu_assembly)
{
  _gpu_assembly = gpu_assembly;
  _system = &system;
  _mesh = &_system->mesh();

  setupVariables();
  setupDofs();

  if (dynamic_cast<NonlinearSystemBase *>(_system))
  {
    setupSparsity();
    checkNodalBCs();
  }

  _qp_solutions.create(MAX_TAG);
  _qp_solutions_grad.create(MAX_TAG);

  _initialized = true;
}

void
GPUSystem::sync(GPUMemcpyKind dir)
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

      _matrices[tag].create(matrix, *this);
      _matrices[tag] = 0;
    }

    _vectors.copy();
    _matrices.copy();
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
GPUSystem::sync(std::set<TagID> tags, GPUMemcpyKind dir)
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

    _vectors.copy();
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
GPUSystem::setActiveVariables(std::set<MooseVariableFieldBase *> vars)
{
  std::set<unsigned int> active_variables;

  for (auto var : vars)
    if (var->sys().number() == _system->number())
      active_variables.insert(var->number());

  _active_variables = active_variables;
}

void
GPUSystem::setActiveVariableTags(std::set<TagID> tags)
{
  if (!_active_variables.size())
    return;

  std::set<TagID> active_variable_tags;

  for (auto tag : tags)
    if (_system->hasVector(tag))
      active_variable_tags.insert(tag);

  _active_variable_tags = active_variable_tags;
}

void
GPUSystem::setActiveResidualTags(std::set<TagID> tags)
{
  std::set<TagID> active_residual_tags;

  for (auto tag : tags)
    if (_system->hasVector(tag))
    {
      active_residual_tags.insert(tag);
      _residual_tag_active[tag] = true;
    }

  _active_residual_tags = active_residual_tags;

  _residual_tag_active.copy();
}

void
GPUSystem::setActiveMatrixTags(std::set<TagID> tags)
{
  std::set<TagID> active_matrix_tags;

  for (auto tag : tags)
    if (_system->hasMatrix(tag))
    {
      active_matrix_tags.insert(tag);
      _matrix_tag_active[tag] = true;
    }

  _active_matrix_tags = active_matrix_tags;

  _matrix_tag_active.copy();
}

void
GPUSystem::project()
{
  for (auto tag : _active_variable_tags)
  {
    if (!_qp_solutions[tag].isAlloc())
      _qp_solutions[tag].create(_mesh->meshSubdomains().size(), _n_vars);

    if (!_qp_solutions_grad[tag].isAlloc())
      _qp_solutions_grad[tag].create(_mesh->meshSubdomains().size(), _n_vars);

    for (auto subdomain : _mesh->meshSubdomains())
    {
      auto sid = _mesh->getGPUSubdomainID(subdomain);

      for (auto var : _active_variables)
      {
        if (!_var_subdomain_active(var, sid))
          continue;

        if (!_qp_solutions[tag](sid, var).isAlloc())
          _qp_solutions[tag](sid, var).createDevice(_gpu_assembly.getNumQps(sid));

        if (!_qp_solutions_grad[tag](sid, var).isAlloc())
          _qp_solutions_grad[tag](sid, var).createDevice(_gpu_assembly.getNumQps(sid));
      }
    }

    _qp_solutions[tag].copy();
    _qp_solutions_grad[tag].copy();
  }

  _qp_solutions.copy();
  _qp_solutions_grad.copy();

  _thread.resize({_gpu_assembly.getMaxQpsPerElem(),
                  _n_elems,
                  _active_variables.size(),
                  _active_variable_tags.size()});

  Kokkos::RangePolicy<Kokkos::IndexType<size_t>> policy(0, _thread.size());
  Kokkos::parallel_for(policy, *this);
  Kokkos::fence();
}

KOKKOS_FUNCTION void
GPUSystem::operator()(const size_t tid) const
{
  auto qp = _thread(tid, 0);
  auto elem = _thread(tid, 1);
  auto var = _active_variables[_thread(tid, 2)];
  auto tag = _active_variable_tags[_thread(tid, 3)];

  auto info = _gpu_assembly.getElementInfo(elem);

  if (qp >= _gpu_assembly.n_qps(info))
    return;

  if (!_var_subdomain_active(var, info.subdomain))
    return;

  auto fe = _var_fe_types[var];
  auto n_dofs = _gpu_assembly.n_dofs(elem, fe);
  auto qp_offset = _gpu_assembly.getQpOffset(info);

  auto & phi = _gpu_assembly.getPhi(info.subdomain, info.type, fe);
  auto & grad_phi = _gpu_assembly.getGradPhi(info.subdomain, info.type, fe);
  auto jac = _gpu_assembly.getJacobian(info.subdomain)(qp_offset + qp);

  Real value = 0;
  Real3 grad = 0;

  for (unsigned int i = 0; i < n_dofs; ++i)
  {
    auto vector = getVectorDofValue(getElemLocalDofIndex(elem, i, var), tag);

    value += vector * phi(i, qp);
    grad += vector * (jac * grad_phi(i, qp));
  }

  getVectorQpValue(info, qp_offset + qp, var, tag) = value;
  getVectorQpGrad(info, qp_offset + qp, var, tag) = grad;
}
