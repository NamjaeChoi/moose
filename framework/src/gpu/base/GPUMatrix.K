//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#ifndef MOOSE_IGNORE_LIBCEED

#include "GPUMatrix.h"

void
GPUMatrix::create(libMesh::SparseMatrix<PetscScalar> & matrix)
{
  auto petsc_matrix = dynamic_cast<libMesh::PetscMatrix<PetscScalar> *>(&matrix);

  mooseAssert(petsc_matrix, "GPUMatrix error: provided matrix is not a PetscMatrix.");

  PetscBool is_host;
  LibmeshPetscCall(MatBoundToCPU(petsc_matrix->mat(), &is_host));

  if (_is_alloc)
    return;

  _is_alloc = true;
}

void
GPUMatrix::destroy()
{
  _matrix = PETSC_NULLPTR;
  _nr = 0;

  _col_idx.destroy();
  _row_idx.destroy();
  _row_ptr.destroy();
  _val.destroy();

  _is_host = false;
  _is_alloc = false;
}

void
GPUMatrix::close()
{
  if (_is_host)
    _val.copy(GPUMemcpyKind::DEVICE_TO_HOST);

  LibmeshPetscCall(
      MatSetValuesCOO(_matrix, _is_host ? _val.host_data() : _val.device_data(), ADD_VALUES));
}

#endif
