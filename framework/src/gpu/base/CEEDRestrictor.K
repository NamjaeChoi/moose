//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#ifndef MOOSE_IGNORE_LIBCEED

#include "CEEDRestrictor.h"

#include "MooseVariableFieldBase.h"
#include "SystemBase.h"

#include "libmesh/system.h"
#include "libmesh/dof_map.h"
#include "libmesh/petsc_vector.h"

void
CEEDRestrictor::createElementRestriction(const CEEDAssembly & assembly,
                                         const std::vector<MooseVariableFieldBase *> & variables,
                                         const std::set<SubdomainID> * blocks,
                                         const std::set<BoundaryID> * boundaries)
{
  auto mesh = assembly.getMesh();

  std::unordered_set<const Elem *> elems;

  if (blocks)
  {
    for (auto elem : mesh->getMesh().active_subdomain_set_elements_ptr_range(*blocks))
    {
      if (boundaries)
      {
        for (auto boundary : *boundaries)
          if (mesh->sideWithBoundaryID(elem, boundary) != libMesh::invalid_uint)
          {
            elems.insert(elem);
            break;
          }
      }
      else
        elems.insert(elem);
    }
  }
  else
  {
    for (auto elem : *mesh->getActiveLocalElementRange())
    {
      if (boundaries)
      {
        for (auto boundary : *boundaries)
          if (mesh->sideWithBoundaryID(elem, boundary) != libMesh::invalid_uint)
          {
            elems.insert(elem);
            break;
          }
      }
      else
        elems.insert(elem);
    }
  }

  _elem_rstr.clear();

  for (auto var : variables)
  {
    auto & sys = var->sys().system();
    auto & dof_map = sys.get_dof_map();
    auto solution = dynamic_cast<PetscVector<Number> *>(sys.current_local_solution.get());

    for (auto type : assembly.getUniqueElemTypes())
      for (auto qrule : assembly.getUniqueQuadratures())
      {
        auto tuple = std::make_tuple(var, type, qrule->get_order());
        auto & rstr = _elem_rstr[tuple];

        qrule->init(type);

        CeedInt num_elem = 0;
        CeedInt elem_size = 0;
        CeedInt num_comp = var->count();

        std::vector<dof_id_type> dof_indices;
        std::vector<CeedInt> offsets;

        for (auto elem : elems)
        {
          if (elem->type() != type)
            continue;

          if (!var->activeOnSubdomain(elem->subdomain_id()))
            continue;

          if (assembly.getQuadratureOrder(elem) != qrule->get_order())
            continue;

          dof_map.dof_indices(elem, dof_indices, var->number());

          for (auto dof : dof_indices)
          {
            auto offset = solution->map_global_to_local_index(dof);
            offsets.push_back(offset);
          }

          num_elem++;
          elem_size = dof_indices.size();
        }

        CeedElemRestrictionCreate(assembly.getCeed(),
                                  num_elem,
                                  elem_size,
                                  num_comp,
                                  1,
                                  sys.n_local_dofs() + dof_map.get_send_list().size(),
                                  CEED_MEM_HOST,
                                  CEED_COPY_VALUES,
                                  offsets.data(),
                                  &rstr);
      }
  }
}

void
CEEDRestrictor::createQuadratureRestriction(const CEEDAssembly & assembly,
                                            const std::vector<MooseVariableFieldBase *> & variables,
                                            const std::set<SubdomainID> * blocks,
                                            const std::set<BoundaryID> * boundaries)
{
  auto mesh = assembly.getMesh();
  CeedInt dim = mesh->dimension();

  std::unordered_set<const Elem *> elems;

  if (blocks)
  {
    for (auto elem : mesh->getMesh().active_subdomain_set_elements_ptr_range(*blocks))
    {
      if (boundaries)
        for (auto boundary : *boundaries)
          if (mesh->sideWithBoundaryID(elem, boundary) != libMesh::invalid_uint)
          {
            elems.insert(elem);
            break;
          }
          else
            elems.insert(elem);
    }
  }
  else
  {
    for (auto elem : *mesh->getActiveLocalElementRange())
    {
      if (boundaries)
        for (auto boundary : *boundaries)
          if (mesh->sideWithBoundaryID(elem, boundary) != libMesh::invalid_uint)
          {
            elems.insert(elem);
            break;
          }
          else
            elems.insert(elem);
    }
  }

  _qp_rstr_interp.clear();
  _qp_rstr_grad.clear();

  for (auto var : variables)
    for (auto type : assembly.getUniqueElemTypes())
      for (auto qrule : assembly.getUniqueQuadratures())
      {
        auto tuple = std::make_tuple(var, type, qrule->get_order());
        auto & rstr_interp = _qp_rstr_interp[tuple];
        auto & rstr_grad = _qp_rstr_grad[tuple];

        qrule->init(type);

        CeedInt num_elem = 0;
        CeedInt elem_size = qrule->n_points();
        CeedInt num_comp = var->count();
        CeedInt elem_idx = 0;

        std::unordered_map<const Elem *, CeedInt> elem_indices;

        for (auto elem : *mesh->getActiveLocalElementRange())
        {
          if (elem->type() != type)
            continue;

          if (!var->activeOnSubdomain(elem->subdomain_id()))
            continue;

          if (assembly.getQuadratureOrder(elem) != qrule->get_order())
            continue;

          elem_indices[elem] = elem_idx++;
        }

        if (!blocks && !boundaries)
        {
          CeedInt strides[3];

          strides[0] = num_comp;
          strides[1] = 1;
          strides[2] = num_comp * elem_size;

          CeedElemRestrictionCreateStrided(assembly.getCeed(),
                                           elem_indices.size(),
                                           elem_size,
                                           num_comp,
                                           elem_indices.size() * elem_size * num_comp,
                                           strides,
                                           &rstr_interp);

          strides[0] = dim * num_comp;
          strides[1] = dim;
          strides[2] = dim * num_comp * elem_size;

          CeedElemRestrictionCreateStrided(assembly.getCeed(),
                                           elem_indices.size(),
                                           elem_size,
                                           num_comp * dim,
                                           elem_indices.size() * elem_size * num_comp * dim,
                                           strides,
                                           &rstr_grad);
        }
        else
        {
          std::vector<CeedInt> offsets_interp;
          std::vector<CeedInt> offsets_grad;

          for (auto elem : elems)
          {
            if (elem->type() != type)
              continue;

            if (!var->activeOnSubdomain(elem->subdomain_id()))
              continue;

            if (assembly.getQuadratureOrder(elem) != qrule->get_order())
              continue;

            for (CeedInt i = 0; i < elem_size; ++i)
            {
              offsets_interp.push_back((elem_indices[elem] * elem_size + i) * num_comp);
              offsets_grad.push_back((elem_indices[elem] * elem_size + i) * num_comp * dim);
            }

            num_elem++;
          }

          CeedElemRestrictionCreate(assembly.getCeed(),
                                    num_elem,
                                    elem_size,
                                    num_comp,
                                    1,
                                    elem_indices.size() * elem_size * num_comp,
                                    CEED_MEM_HOST,
                                    CEED_COPY_VALUES,
                                    offsets_interp.data(),
                                    &rstr_interp);

          CeedElemRestrictionCreate(assembly.getCeed(),
                                    num_elem,
                                    elem_size,
                                    num_comp * dim,
                                    1,
                                    elem_indices.size() * elem_size * num_comp * dim,
                                    CEED_MEM_HOST,
                                    CEED_COPY_VALUES,
                                    offsets_grad.data(),
                                    &rstr_grad);
        }
      }
}

#endif
