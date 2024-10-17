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

#include "GPUArray.h"

#include "libmesh/petsc_matrix.h"

class GPUMatrix
{
private:
  // The PETSc matrix
  Mat _matrix = PETSC_NULLPTR;
  // Number of rows
  PetscCount _nr = 0;
  // Column index vector
  GPUArray<PetscInt> _col_idx;
  // Row index vector
  GPUArray<PetscInt> _row_idx;
  // Row pointer vector
  GPUArray<PetscInt> _row_ptr;
  // Nonzero value vector
  GPUArray<PetscScalar> _val;
  // Flag whether the PETSc matrix is a host matrix
  bool _is_host = false;
  // Flag whether the matrix was allocated
  bool _is_alloc = false;

#ifdef MOOSE_GPU_SCOPE
public:
  // Destructor
  ~GPUMatrix() { destroy(); }

public:
  // Whether the matrix was allocated
  bool isAlloc() { return _is_alloc; }
  // Create from libMesh PetscMatrix
  void create(libMesh::SparseMatrix<PetscScalar> & matrix);
  // Free all the matrix data
  void destroy();
  // Copy from device buffer to PETSc matrix
  void close();

public:
  // Scalar assignment operator
  void operator=(const PetscScalar & scalar) { _val = scalar; }
#endif
};

#endif
