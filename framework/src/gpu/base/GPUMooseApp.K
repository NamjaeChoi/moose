//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "GPUHeader.h"

#include "MooseApp.h"

void
MooseApp::initializeGPUs()
{
  _num_GPUs = Kokkos::num_devices();

  if (!_num_GPUs)
    return;

  Kokkos::InitializationSettings settings;

  // Create a local communicator defined at each shared memory node
  Parallel::Communicator local_comm;
  _comm->split_by_type(MPI_COMM_TYPE_SHARED, processor_id(), MPI_INFO_NULL, local_comm);

  // The number of processes in each node is usually larger than the number of
  // GPU devices in the node, so multiple processes share the same GPU
  _my_GPU_id = local_comm.rank() % _num_GPUs;

  // Override the default GPU ID
  settings.set_device_id(_my_GPU_id);

  Kokkos::initialize(settings);

  std::atexit(Kokkos::finalize);
}
