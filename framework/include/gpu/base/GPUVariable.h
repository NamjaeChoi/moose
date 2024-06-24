//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#ifdef MOOSE_GPU_SCOPE
#include "GPUHeader.h"
#endif

#include "MooseTypes.h"

class Coupleable;

class GPUVariable
{
  friend class Coupleable;

private:
  // Whether this variable was coupled
  bool _coupled = true;
  // The number of components
  unsigned int _components = 0;
  // The system number
  unsigned int _sys = -1;
  // The vector tag ID
  TagID _tag = Moose::INVALID_TAG_ID;
  // Variable numbers
  GPUArray<unsigned int> _vars;
  // Default value of each component
  GPUArray<Real> _default_values;

public:
#ifdef MOOSE_GPU_SCOPE
  KOKKOS_FUNCTION bool coupled() const { return _coupled; }
  KOKKOS_FUNCTION unsigned int components() { return _components; }
  KOKKOS_FUNCTION unsigned int sys() const { return _sys; }
  KOKKOS_FUNCTION TagID tag() const { return _tag; }
  KOKKOS_FUNCTION unsigned int var(unsigned int comp = 0) const { return _vars[comp]; }
  KOKKOS_FUNCTION Real value(unsigned int comp = 0) const { return _default_values[comp]; }
#endif
};
