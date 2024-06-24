//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "GPUAssembly.h"

#include "MooseMesh.h"
#include "FEProblemBase.h"
#include "NonlinearSystemBase.h"
#include "AuxiliarySystem.h"
#include "Assembly.h"

void
GPUAssembly::initQuadrature(FEProblemBase & problem)
{
  _problem = &problem;
  _mesh = &_problem->mesh();
  _assembly = &_problem->assembly(0, 0);
  _n_subdomains = _mesh->meshSubdomains().size();

  _n_qps.create(_n_subdomains);
  _n_qps_face.create(_n_subdomains);
  _qp_offset.create(_n_subdomains);
  _qp_offset_face.create(_n_subdomains);

  _max_qps_per_elem.create(_n_subdomains + 1);
  _max_qps_per_elem = 0;

  for (auto subdomain : _mesh->meshSubdomains())
  {
    auto sid = _mesh->getGPUSubdomainID(subdomain);
    auto & map = _mesh->getGPUSubdomainLocalElementMap(subdomain);

    _n_qps[sid].create(map.size());
    _n_qps_face[sid].create(_mesh->getMaxSidesPerElem(), map.size());
    _n_qps_face[sid] = 0;

    _qp_offset[sid].create(map.size() + 1);
    _qp_offset_face[sid].create(_mesh->getMaxSidesPerElem(), map.size() + 1);
    _qp_offset_face[sid] = 0;

    for (auto & [elem, eid] : map)
    {
      _assembly->setCurrentSubdomainID(subdomain);
      _assembly->reinit(elem);

      auto qrule = _assembly->qRule();

      _n_qps[sid][eid] = qrule->n_points();
      _qp_offset[sid][eid] = qrule->n_points();

      for (unsigned int side = 0; side < elem->n_sides(); ++side)
      {
        _assembly->reinit(elem, side);

        auto qrule = _assembly->qRuleFace();

        _n_qps_face[sid](side, eid) = qrule->n_points();
        _qp_offset_face[sid](side, eid) = qrule->n_points();
      }

      _max_qps_per_elem[sid] = std::max(_max_qps_per_elem[sid], _n_qps[sid][eid]);
    }

    _max_qps_per_elem.last() = std::max(_max_qps_per_elem.last(), _max_qps_per_elem[sid]);

    std::exclusive_scan(_qp_offset[sid].begin(), _qp_offset[sid].end(), _qp_offset[sid].begin(), 0);
    std::exclusive_scan(
        _qp_offset_face[sid].begin(), _qp_offset_face[sid].end(), _qp_offset_face[sid].begin(), 0);
  }

  _n_qps.copyNested();
  _n_qps_face.copyNested();
  _qp_offset.copyNested();
  _qp_offset_face.copyNested();
  _max_qps_per_elem.copy();
}

void
GPUAssembly::initElement()
{
  // Generate the list of unique FE types

  std::set<FEType> fe_types;

  auto getFETypes = [&](System & system)
  {
    for (unsigned int var = 0; var < system.n_vars(); var++)
      fe_types.insert(system.variable_type(var));
  };

  for (unsigned int nl = 0; nl < _problem->numNonlinearSystems(); ++nl)
    getFETypes(_problem->getNonlinearSystemBase(nl).system());

  getFETypes(_problem->getAuxiliarySystem().system());

  for (auto & fet : fe_types)
    _fe_type_map[fet] = _fe_type_map.size();

  // Cache element information and shape data

  auto n_elems = _mesh->getGPULocalElementMap().size();
  auto n_elem_types = _mesh->getGPUElementTypeMap().size();

  _elem_info.create(n_elems);
  _elem_neighbor.create(_mesh->getMaxSidesPerElem(), n_elems);
  _elem_neighbor = -1;

  _J.create(_n_subdomains);
  _J_face.create(_n_subdomains);
  _JxW_coord.create(_n_subdomains);
  _JxW_coord_face.create(_n_subdomains);

  _phi.create(_n_subdomains, n_elem_types, _fe_type_map.size());
  _phi_face.create(_n_subdomains, n_elem_types, _fe_type_map.size());
  _grad_phi.create(_n_subdomains, n_elem_types, _fe_type_map.size());
  _grad_phi_face.create(_n_subdomains, n_elem_types, _fe_type_map.size());

  _n_dofs.create(n_elems, _fe_type_map.size());
  _n_dofs = 0;

  for (auto subdomain : _mesh->meshSubdomains())
  {
    auto sid = _mesh->getGPUSubdomainID(subdomain);

    for (auto & [fe_type, fe_idx] : _fe_type_map)
      for (unsigned int t = 0; t < n_elem_types; ++t)
      {
        _phi(sid, t, fe_idx).create(MAX_DOF, _max_qps_per_elem[sid]);
        _phi_face(sid, t, fe_idx)
            .create(MAX_DOF, _max_qps_per_elem[sid], _mesh->getMaxSidesPerElem());
        _grad_phi(sid, t, fe_idx).create(MAX_DOF, _max_qps_per_elem[sid]);
        _grad_phi_face(sid, t, fe_idx)
            .create(MAX_DOF, _max_qps_per_elem[sid], _mesh->getMaxSidesPerElem());

        _phi(sid, t, fe_idx) = 0;
        _phi_face(sid, t, fe_idx) = 0;
        _grad_phi(sid, t, fe_idx) = Real3(0);
        _grad_phi_face(sid, t, fe_idx) = Real3(0);
      }

    _J[sid].create(_qp_offset[sid].last());
    _J_face[sid].create(_qp_offset_face[sid].last());
    _JxW_coord[sid].create(_qp_offset[sid].last());
    _JxW_coord_face[sid].create(_qp_offset_face[sid].last());

    _J[sid] = Real33(0);
    _J_face[sid] = Real33(0);
    _JxW_coord[sid] = 0;
    _JxW_coord_face[sid] = 0;
  }

  for (auto subdomain : _mesh->meshSubdomains())
  {
    auto sid = _mesh->getGPUSubdomainID(subdomain);
    auto & map = _mesh->getGPUSubdomainLocalElementMap(subdomain);

    for (auto & [elem, eid] : map)
    {
      auto id = _mesh->getGPUElementID(elem);
      auto elem_type = _mesh->getGPUElementTypeID(elem);

      _elem_info[id].type = elem_type;
      _elem_info[id].id = id;
      _elem_info[id].local_id = eid;
      _elem_info[id].subdomain = sid;

      _assembly->setCurrentSubdomainID(elem->subdomain_id());
      _assembly->reinit(elem);

      auto fe = _assembly->getFE(_fe_type_map.begin()->first, elem->dim());
      auto & JxW = fe->get_JxW();
      auto & dxidx = fe->get_dxidx();
      auto & dxidy = fe->get_dxidy();
      auto & dxidz = fe->get_dxidz();
      auto & detadx = fe->get_detadx();
      auto & detady = fe->get_detady();
      auto & detadz = fe->get_detadz();
      auto & dzetadx = fe->get_dzetadx();
      auto & dzetady = fe->get_dzetady();
      auto & dzetadz = fe->get_dzetadz();

      auto offset = _qp_offset[sid][eid];

      for (unsigned int qp = 0; qp < JxW.size(); ++qp)
      {
        auto & J = _J[sid][offset + qp];

        if (elem->dim() >= 1)
        {
          J(0, 0) = dxidx[qp];
        }
        if (elem->dim() >= 2)
        {
          J(0, 1) = detadx[qp];
          J(1, 0) = dxidy[qp];
          J(1, 1) = detady[qp];
        }
        if (elem->dim() == 3)
        {
          J(0, 2) = dzetadx[qp];
          J(1, 2) = dzetady[qp];
          J(2, 0) = dxidz[qp];
          J(2, 1) = detadz[qp];
          J(2, 2) = dzetadz[qp];
        }

        _JxW_coord[sid][offset + qp] = JxW[qp] * _assembly->coordTransformation()[qp];
      }

      for (auto & [fe_type, fe_idx] : _fe_type_map)
      {
        auto fe = _assembly->getFE(fe_type, elem->dim());
        auto & phi = fe->get_phi();
        auto & dphidxi = fe->get_dphidxi();
        auto & dphideta = fe->get_dphideta();
        auto & dphidzeta = fe->get_dphidzeta();

        _n_dofs(id, fe_idx) = phi.size();

        for (unsigned int i = 0; i < phi.size(); ++i)
          for (unsigned int qp = 0; qp < phi[i].size(); ++qp)
          {
            _phi(sid, elem_type, fe_idx)(i, qp) = phi[i][qp];

            if (elem->dim() >= 1)
              _grad_phi(sid, elem_type, fe_idx)(i, qp).x = dphidxi[i][qp];
            if (elem->dim() >= 2)
              _grad_phi(sid, elem_type, fe_idx)(i, qp).y = dphideta[i][qp];
            if (elem->dim() == 3)
              _grad_phi(sid, elem_type, fe_idx)(i, qp).z = dphidzeta[i][qp];
          }
      }

      for (unsigned int side = 0; side < elem->n_sides(); ++side)
      {
        // if (auto neighbor = elem->neighbor_ptr(side))
        //   _elem_neighbor(side, eid) = _mesh->getGPUElementID(neighbor);

        _assembly->reinit(elem, side);

        auto fe = _assembly->getFEFace(_fe_type_map.begin()->first, elem->dim());
        auto & JxW = fe->get_JxW();
        auto & dxidx = fe->get_dxidx();
        auto & dxidy = fe->get_dxidy();
        auto & detadx = fe->get_detadx();
        auto & detady = fe->get_detady();

        auto offset = _qp_offset_face[sid](side, eid);

        for (unsigned int qp = 0; qp < JxW.size(); ++qp)
        {
          auto & J = _J_face[sid][offset + qp];

          if (elem->dim() >= 2)
          {
            J(0, 0) = dxidx[qp];
          }
          if (elem->dim() == 3)
          {
            J(0, 1) = detadx[qp];
            J(1, 0) = dxidy[qp];
            J(1, 1) = detady[qp];
          }

          _JxW_coord_face[sid][offset + qp] = JxW[qp] * _assembly->coordTransformation()[qp];
        }

        for (auto & [fe_type, fe_idx] : _fe_type_map)
        {
          auto fe = _assembly->getFEFace(fe_type, elem->dim());
          auto & phi = fe->get_phi();
          auto & dphidxi = fe->get_dphidxi();
          auto & dphideta = fe->get_dphideta();

          for (unsigned int i = 0; i < phi.size(); ++i)
            for (unsigned int qp = 0; qp < phi[i].size(); ++qp)
            {
              _phi_face(sid, elem_type, fe_idx)(i, qp, side) = phi[i][qp];

              if (elem->dim() >= 2)
                _grad_phi_face(sid, elem_type, fe_idx)(i, qp, side).x = dphidxi[i][qp];
              if (elem->dim() == 3)
                _grad_phi_face(sid, elem_type, fe_idx)(i, qp, side).y = dphideta[i][qp];
            }
        }
      }
    }
  }

  _elem_info.copy();
  _elem_neighbor.copy();

  _J.copyNested();
  _J_face.copyNested();
  _JxW_coord.copyNested();
  _JxW_coord_face.copyNested();

  _phi.copyNested();
  _phi_face.copyNested();
  _grad_phi.copyNested();
  _grad_phi_face.copyNested();

  _n_dofs.copy();
}

void
GPUAssembly::initNeighbor()
{
  _phi_neighbor.create(_n_subdomains, _fe_type_map.size());
  _grad_phi_neighbor.create(_n_subdomains, _fe_type_map.size());

  for (auto subdomain : _mesh->meshSubdomains())
  {
    auto sid = _mesh->getGPUSubdomainID(subdomain);
    auto & map = _mesh->getGPUSubdomainLocalElementMap(subdomain);

    std::vector<unsigned int> max_dofs_neighbor(_fe_type_map.size());

    for (auto & [elem, eid] : map)
      for (unsigned int side = 0; side < elem->n_sides(); ++side)
        if (auto neighbor = elem->neighbor_ptr(side))
        {
          auto neighbor_id = _mesh->getGPUElementID(neighbor);

          for (auto & [fe, fe_type] : _fe_type_map)
            max_dofs_neighbor[fe_type] =
                std::max(max_dofs_neighbor[fe_type], _n_dofs(neighbor_id, fe_type));
        }

    for (auto & [fe, fe_type] : _fe_type_map)
    {
      _phi_neighbor(sid, fe_type).create(max_dofs_neighbor[fe_type], _qp_offset_face[sid].last());
      _grad_phi_neighbor(sid, fe_type)
          .create(max_dofs_neighbor[fe_type], _qp_offset_face[sid].last());

      _phi_neighbor(sid, fe_type) = 0;
      _grad_phi_neighbor(sid, fe_type) = Real3(0);
    }
  }

  for (auto subdomain : _mesh->meshSubdomains())
  {
    auto sid = _mesh->getGPUSubdomainID(subdomain);
    auto & map = _mesh->getGPUSubdomainLocalElementMap(subdomain);

    for (auto & [elem, eid] : map)
      for (unsigned int side = 0; side < elem->n_sides(); ++side)
      {
        auto offset = _qp_offset_face[sid](side, eid);

        if (auto neighbor = elem->neighbor_ptr(side))
        {
          _assembly->setCurrentNeighborSubdomainID(neighbor->subdomain_id());
          _assembly->reinitElemAndNeighbor(
              elem, side, neighbor, neighbor->which_neighbor_am_i(elem));

          auto qrule = _assembly->qRuleNeighbor();

          for (auto & [fe, fe_type] : _fe_type_map)
          {
            auto & phi = _assembly->fePhiFaceNeighbor<Real>(fe);
            auto & grad_phi = _assembly->feGradPhiFaceNeighbor<Real>(fe);

            for (unsigned int i = 0; i < phi.size(); ++i)
              for (unsigned int qp = 0; qp < qrule->n_points(); ++qp)
              {
                _phi_neighbor(sid, fe_type)(i, offset + qp) = phi[i][qp];
                _grad_phi_neighbor(sid, fe_type)(i, offset + qp) = grad_phi[i][qp];
              }
          }
        }
      }
  }

  _phi_neighbor.copyNested();
  _grad_phi_neighbor.copyNested();
}
