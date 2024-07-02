//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "GPUMaterial.h"

/**
 * Simple material with properties set as constants.
 */
class GPUHeatConductionMaterial final : public GPUMaterial<GPUHeatConductionMaterial>
{
public:
  static InputParameters validParams();

  GPUHeatConductionMaterial(const InputParameters & parameters);

  KOKKOS_FUNCTION void computeQpProperties(const Datum & datum, unsigned int qp) const
  {
    datum.setProperty(_thermal_conductivity, qp) = _my_thermal_conductivity;
    datum.setProperty(_thermal_conductivity_dT, qp) = 0;
    datum.setProperty(_specific_heat, qp) = _my_specific_heat;
    datum.setProperty(_specific_heat_dT, qp) = 0;
  }

private:
  const bool _has_temp;
  GPUVariable _temperature;

  const Real _my_thermal_conductivity;
  const Real _my_specific_heat;

  GPUMaterialProperty<Real> _thermal_conductivity;
  GPUMaterialProperty<Real> _thermal_conductivity_dT;

  GPUMaterialProperty<Real> _specific_heat;
  GPUMaterialProperty<Real> _specific_heat_dT;

  /// Minimum temperature, below which temperature is "clipped" before evaluating functions
  const Real _min_T;
};
