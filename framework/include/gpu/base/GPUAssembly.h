//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "GPUTypes.h"

#include "libmesh/elem_range.h"
#include "libmesh/fe_base.h"
#include "libmesh/fe_type.h"

class FEProblemBase;
class MooseMesh;
class Assembly;

struct GPUElementInfo
{
  // Element type ID
  unsigned int type;
  // Element ID
  dof_id_type id;
  // Subdomain-local element ID
  dof_id_type local_id;
  // Subdomain ID
  SubdomainID subdomain;
};

class GPUAssembly
{
public:
  // Initialize assembly
  void initQuadrature(FEProblemBase & problem);
  void initElement();
  void initNeighbor();

#ifdef MOOSE_GPU_SCOPE
  // Get FE type number
  auto getFETypeNum(FEType type) const { return _fe_type_map.at(type); }
  // Get the element information
  KOKKOS_FUNCTION auto getElementInfo(dof_id_type elem) const { return _elem_info[elem]; }
  // Get the neighbor local element ID
  KOKKOS_FUNCTION auto getNeighbor(dof_id_type elem, unsigned int side) const
  {
    return _elem_neighbor(side, elem);
  }
  // Get the number of quadrature points
  KOKKOS_FUNCTION auto getMaxQpsPerElem() const { return _max_qps_per_elem.last(); }
  KOKKOS_FUNCTION auto getMaxQpsPerElem(SubdomainID subdomain) const
  {
    return _max_qps_per_elem[subdomain];
  }
  KOKKOS_FUNCTION auto getNumQps(SubdomainID subdomain) const
  {
    return _qp_offset[subdomain].last();
  }
  KOKKOS_FUNCTION auto getNumFaceQps(SubdomainID subdomain) const
  {
    return _qp_offset_face[subdomain].last();
  }
  // Get the quadrature point offset
  KOKKOS_FUNCTION auto getQpOffset(GPUElementInfo info) const
  {
    return _qp_offset[info.subdomain][info.local_id];
  }
  KOKKOS_FUNCTION auto getQpFaceOffset(GPUElementInfo info, unsigned int side) const
  {
    return _qp_offset_face[info.subdomain](side, info.local_id);
  }
  // Get shape data
  KOKKOS_FUNCTION const auto &
  getPhi(SubdomainID subdomain, unsigned int elem_type, unsigned int fe_type) const
  {
    return _phi(subdomain, elem_type, fe_type);
  }
  KOKKOS_FUNCTION const auto &
  getPhiFace(SubdomainID subdomain, unsigned int elem_type, unsigned int fe_type) const
  {
    return _phi_face(subdomain, elem_type, fe_type);
  }
  KOKKOS_FUNCTION const auto & getPhiNeighbor(SubdomainID subdomain, unsigned int fe_type) const
  {
    return _phi_neighbor(subdomain, fe_type);
  }
  KOKKOS_FUNCTION const auto &
  getGradPhi(SubdomainID subdomain, unsigned int elem_type, unsigned int fe_type) const
  {
    return _grad_phi(subdomain, elem_type, fe_type);
  }
  KOKKOS_FUNCTION const auto &
  getGradPhiFace(SubdomainID subdomain, unsigned int elem_type, unsigned int fe_type) const
  {
    return _grad_phi_face(subdomain, elem_type, fe_type);
  }
  KOKKOS_FUNCTION const auto & getGradPhiNeighbor(SubdomainID subdomain, unsigned int fe_type) const
  {
    return _grad_phi_neighbor(subdomain, fe_type);
  }
  KOKKOS_FUNCTION auto n_dofs(dof_id_type elem, unsigned int fe_type) const
  {
    return _n_dofs(elem, fe_type);
  }
  // Get transformation data
  KOKKOS_FUNCTION const auto & getJacobian(SubdomainID subdomain) const { return _J[subdomain]; }
  KOKKOS_FUNCTION const auto & getJacobianFace(SubdomainID subdomain) const
  {
    return _J_face[subdomain];
  }
  KOKKOS_FUNCTION const auto & getJxWCoord(SubdomainID subdomain) const
  {
    return _JxW_coord[subdomain];
  }
  KOKKOS_FUNCTION const auto & getJxWCoordFace(SubdomainID subdomain) const
  {
    return _JxW_coord_face[subdomain];
  }
  KOKKOS_FUNCTION auto n_qps(GPUElementInfo info) const
  {
    return _n_qps[info.subdomain][info.local_id];
  }
  KOKKOS_FUNCTION auto n_qps_face(GPUElementInfo info, unsigned int side) const
  {
    return _n_qps_face[info.subdomain](side, info.local_id);
  }
#endif

private:
  // Pointer to MOOSE problem
  FEProblemBase * _problem = nullptr;
  // Pointer to MOOSE mesh
  const MooseMesh * _mesh = nullptr;
  // Pointer to MOOSE assembly
  Assembly * _assembly = nullptr;
  // Number of subdomains
  unsigned int _n_subdomains = 0;

private:
  // Unique FE type map
  std::map<FEType, unsigned int> _fe_type_map;
  // Element information
  GPUArray<GPUElementInfo> _elem_info;
  // Neighbor elements of each element
  GPUArray2D<dof_id_type> _elem_neighbor;
  // Quadrature point offset of each element
  GPUArray<GPUArray<dof_id_type>> _qp_offset;
  GPUArray<GPUArray2D<dof_id_type>> _qp_offset_face;
  // Number of quadrature points of each element
  GPUArray<GPUArray<unsigned int>> _n_qps;
  GPUArray<GPUArray2D<unsigned int>> _n_qps_face;
  GPUArray<unsigned int> _max_qps_per_elem;
  // Transformation data of each element
  GPUArray<GPUArray<Real33>> _J;
  GPUArray<GPUArray<Real33>> _J_face;
  GPUArray<GPUArray<Real>> _JxW_coord;
  GPUArray<GPUArray<Real>> _JxW_coord_face;
  // Shape data
  GPUArray3D<GPUArray2D<Real>> _phi;
  GPUArray3D<GPUArray3D<Real>> _phi_face;
  GPUArray2D<GPUArray2D<Real>> _phi_neighbor;
  GPUArray3D<GPUArray2D<Real3>> _grad_phi;
  GPUArray3D<GPUArray3D<Real3>> _grad_phi_face;
  GPUArray2D<GPUArray2D<Real3>> _grad_phi_neighbor;
  GPUArray2D<unsigned int> _n_dofs;
};
