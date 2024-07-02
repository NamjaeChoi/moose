//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "GPUAssembly.h"

#include "MooseMesh.h"

class GPUMaterialPropertyStorage;

template <typename T, unsigned int dimension = 0>
class GPUPropertyBase;

class GPUMaterialPropertyBase
{
  friend class GPUMaterialPropertyStorage;

protected:
  // The property name
  std::string _name;
  // Data type
  std::string _type;
  // Dimension
  unsigned int _dim = 0;
  // The property ID
  unsigned int _id = -1;
  // Whether this property has a default value
  bool _default = false;

public:
  // Get the property name
  std::string name() { return _name; }
  // Get the data atype
  std::string type() { return _type; }
  // Get the dimension
  unsigned int dim() { return _dim; }

#ifdef MOOSE_GPU_SCOPE
private:
  // Allocate data
  virtual void allocate(const MooseMesh & mesh,
                        const GPUAssembly & assembly,
                        const std::set<SubdomainID> & subdomains,
                        const std::string & prop_name,
                        const unsigned int id,
                        const std::vector<unsigned int> & dims,
                        const bool bnd) = 0;

public:
  // Whether this material property is valid
  KOKKOS_FUNCTION operator bool() const { return _id != -1 || _default; }
#endif
};

template <typename T, unsigned int dimension = 0>
class GPUMaterialProperty : public GPUMaterialPropertyBase
{
  friend class GPUPropertyBase<T, dimension>;

public:
  GPUMaterialProperty() {}
  // Constructor for default material property
  GPUMaterialProperty(const T value)
  {
    _default = true;
    _value = value;
  }

private:
  // Reference property
  const GPUMaterialProperty<T, dimension> * _reference = nullptr;
  // Data array
  GPUArray<GPUArray<T, dimension + 1>> _data;
  // Default value
  T _value;

#ifdef MOOSE_GPU_SCOPE
private:
  // Allocate data
  virtual void allocate(const MooseMesh & mesh,
                        const GPUAssembly & assembly,
                        const std::set<SubdomainID> & subdomains,
                        const std::string & prop_name,
                        const unsigned int id,
                        const std::vector<unsigned int> & dims,
                        const bool bnd) override
  {
    // Let MaterialData::declareGPUProperty() error out
    if (dims.size() != dimension)
      return;

    _name = prop_name;
    _type = MooseUtils::prettyCppType<T>();
    _dim = dimension;
    _id = id;

    if (!_data.isAlloc())
      _data.create(mesh.meshSubdomains().size());

    for (auto subdomain : subdomains)
    {
      auto sid = mesh.getGPUSubdomainID(subdomain);

      std::vector<uint64_t> n;

      for (unsigned int i = 0; i < dimension; ++i)
        n.push_back(dims[i]);

      n.push_back(bnd ? assembly.getNumFaceQps(sid) : assembly.getNumQps(sid));

      if (!_data[sid].isAlloc())
        _data[sid].createDevice(n);
    }

    _data.copy();
  }

public:
  // Copy constructor
  GPUMaterialProperty(const GPUMaterialProperty<T, dimension> & property)
  {
    if (!property._reference)
    {
      *this = property;
      this->_reference = &property;
    }
    else
    {
      *this = *property._reference;
      this->_reference = property._reference;
    }
  }
  // Get the mirror property
  GPUMaterialProperty<T, dimension> mirror() { return GPUMaterialProperty<T, dimension>(*this); }
#endif
};

#define usingGPUPropertyBaseMembers(T, dimension)                                                  \
  using GPUPropertyBase<T, dimension>::_qp;                                                        \
  using GPUPropertyBase<T, dimension>::_data;                                                      \
  using GPUPropertyBase<T, dimension>::_value

template <typename T, unsigned int dimension>
class GPUPropertyBase
{
protected:
  // Current quadrature point
  const dof_id_type _qp;
  // Data array
  const GPUArray<T, dimension + 1> * _data;
  // Default value
  const T & _value;

#ifdef MOOSE_GPU_SCOPE
public:
  KOKKOS_FUNCTION GPUPropertyBase(const GPUMaterialProperty<T, dimension> & property,
                                  SubdomainID sid,
                                  dof_id_type qp)
    : _qp(qp), _data(property._default ? nullptr : &property._data[sid]), _value(property._value)
  {
  }
  // Get the size of each dimension
  KOKKOS_FUNCTION uint64_t n(unsigned int dim) { return _data->n(dim); }
#endif
};

template <typename T, unsigned int dimension>
class GPUProperty
{
};

template <typename T>
class GPUProperty<T, 0> : public GPUPropertyBase<T, 0>
{
  usingGPUPropertyBaseMembers(T, 0);

#ifdef MOOSE_GPU_SCOPE
public:
  KOKKOS_FUNCTION
  GPUProperty(const GPUMaterialProperty<T, 0> & property, SubdomainID sid, dof_id_type qp)
    : GPUPropertyBase<T, 0>(property, sid, qp)
  {
  }
  KOKKOS_FUNCTION operator const T &() const { return _data ? (*_data)(_qp) : _value; }
  KOKKOS_FUNCTION void operator=(const T & data) { (*_data)(_qp) = data; }
#endif
};

template <typename T>
class GPUProperty<T, 1> : public GPUPropertyBase<T, 1>
{
  usingGPUPropertyBaseMembers(T, 1);

#ifdef MOOSE_GPU_SCOPE
public:
  KOKKOS_FUNCTION
  GPUProperty(const GPUMaterialProperty<T, 1> & property, SubdomainID sid, dof_id_type qp)
    : GPUPropertyBase<T, 1>(property, sid, qp)
  {
  }
  KOKKOS_FUNCTION T & operator()(unsigned int x) { return (*_data)(x, _qp); }
  KOKKOS_FUNCTION const T & operator()(unsigned int x) const
  {
    return _data ? (*_data)(x, _qp) : _value;
  }
#endif
};

template <typename T>
class GPUProperty<T, 2> : public GPUPropertyBase<T, 2>
{
  usingGPUPropertyBaseMembers(T, 2);

#ifdef MOOSE_GPU_SCOPE
public:
  KOKKOS_FUNCTION
  GPUProperty(const GPUMaterialProperty<T, 2> & property, SubdomainID sid, dof_id_type qp)
    : GPUPropertyBase<T, 2>(property, sid, qp)
  {
  }
  KOKKOS_FUNCTION T & operator()(unsigned int x, unsigned int y) { return (*_data)(x, y, _qp); }
  KOKKOS_FUNCTION const T & operator()(unsigned int x, unsigned int y) const
  {
    return _data ? (*_data)(x, y, _qp) : _value;
  }
#endif
};

template <typename T>
class GPUProperty<T, 3> : public GPUPropertyBase<T, 3>
{
  usingGPUPropertyBaseMembers(T, 3);

#ifdef MOOSE_GPU_SCOPE
public:
  KOKKOS_FUNCTION
  GPUProperty(const GPUMaterialProperty<T, 3> & property, SubdomainID sid, dof_id_type qp)
    : GPUPropertyBase<T, 3>(property, sid, qp)
  {
  }
  KOKKOS_FUNCTION T & operator()(unsigned int x, unsigned int y, unsigned int z)
  {
    return (*_data)(x, y, z, _qp);
  }
  KOKKOS_FUNCTION const T & operator()(unsigned int x, unsigned int y, unsigned int z) const
  {
    return _data ? (*_data)(x, y, z, _qp) : _value;
  }
#endif
};

template <typename T>
class GPUProperty<T, 4> : public GPUPropertyBase<T, 4>
{
  usingGPUPropertyBaseMembers(T, 4);

#ifdef MOOSE_GPU_SCOPE
public:
  KOKKOS_FUNCTION
  GPUProperty(const GPUMaterialProperty<T, 4> & property, SubdomainID sid, dof_id_type qp)
    : GPUPropertyBase<T, 4>(property, sid, qp)
  {
  }
  KOKKOS_FUNCTION T & operator()(unsigned int x, unsigned int y, unsigned int z, unsigned int w)
  {
    return (*_data)(x, y, z, w, _qp);
  }
  KOKKOS_FUNCTION const T &
  operator()(unsigned int x, unsigned int y, unsigned int z, unsigned int w) const
  {
    return _data ? (*_data)(x, y, z, w, _qp) : _value;
  }
#endif
};
