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

#include "CEEDKernelBase.h"

#include "MooseVariableInterface.h"

#define CEED_KERNEL_TEMPLATE(name, parent, symbol)                                                 \
  template <typename symbol>                                                                       \
  class name : public parent<symbol>                                                               \
  {
#define CEED_KERNEL_TEMPLATE_END(name)                                                             \
  }                                                                                                \
  ;

#define CEED_KERNEL(name, parent)                                                                  \
  class name final : public parent<name>                                                           \
  {                                                                                                \
  private:                                                                                         \
    static CEEDQFunctionPointer _qf;                                                               \
    static const char * _qf_loc;                                                                   \
                                                                                                   \
  public:                                                                                          \
    auto getQFunction() { return _qf; }                                                            \
    auto getQFunctionPath() { return _qf_loc; }
#define CEED_KERNEL_END(name)                                                                      \
  }                                                                                                \
  ;                                                                                                \
                                                                                                   \
  CEED_QFUNCTION(name##_qf)                                                                        \
  (void * ctx, const CeedInt Q, const CeedScalar * const * in, CeedScalar * const * out)           \
  {                                                                                                \
    auto kernel = static_cast<name *>(ctx);                                                        \
    kernel->computeResidualInternal(Q, in, out);                                                   \
    return 0;                                                                                      \
  }                                                                                                \
                                                                                                   \
  CEEDQFunctionPointer name::_qf = name##_qf;                                                      \
  const char * name::_qf_loc = name##_qf_loc;

template <typename Kernel>
class CEEDKernel : public CEEDKernelBase, public MooseVariableInterface<Real>
{
public:
  static InputParameters validParams()
  {
    InputParameters params = CEEDKernelBase::validParams();
    params.registerBase("Kernel");
    return params;
  }

  // Constructor
  CEEDKernel(const InputParameters & parameters)
    : CEEDKernelBase(parameters),
      MooseVariableInterface<Real>(this,
                                   false,
                                   "variable",
                                   Moose::VarKindType::VAR_SOLVER,
                                   Moose::VarFieldType::VAR_FIELD_STANDARD),
      _var(*mooseVariable())
  {
    addMooseVariableDependency(mooseVariable());
  }

  // Copy constructor
  CEEDKernel(const CEEDKernel<Kernel> & object)
    : CEEDKernelBase(object),
      MooseVariableInterface<Real>(this,
                                   false,
                                   "variable",
                                   Moose::VarKindType::VAR_SOLVER,
                                   Moose::VarFieldType::VAR_FIELD_STANDARD,
                                   false),
      _var(object._var)
  {
  }

  virtual void initialSetup() override
  {
    initializeCEEDBlockRestrictable(_assembly, getMooseVariableDependencies());

    std::set<TagID> tags(getFEVariableCoupleableVectorTags());

    for (auto tag : _fe_problem.getVectorTags(Moose::VECTOR_TAG_SOLUTION))
      tags.insert(tag._id);

    CeedQFunctionContextCreate(_ceed, &_ctx);

    for (auto type : _assembly.getUniqueElemTypes())
      for (auto qrule : _assembly.getUniqueQuadratures())
      {
        /// Set QFunction

        auto kernel = static_cast<Kernel *>(this);
        auto pair = std::make_pair(type, qrule->get_order());
        auto & qf = _functions[pair];

        CeedQFunctionCreateInterior(
            _ceed, 1, kernel->getQFunction(), kernel->getQFunctionPath(), &qf);
        CeedQFunctionSetContext(qf, _ctx);

        for (auto var : getMooseVariableDependencies())
          for (auto tag : tags)
          {
            std::string name = var->name() + "_tag_" + std::to_string(tag);

            CeedQFunctionAddInput(qf, name.c_str(), var->count(), CEED_EVAL_NONE);
            CeedQFunctionAddInput(
                qf, (name + "_grad").c_str(), var->count() * _mesh.dimension(), CEED_EVAL_NONE);
          }

        CeedQFunctionAddOutput(qf, "residual", _var.count(), CEED_EVAL_INTERP);

        /// Set operator

        auto tuple = std::make_tuple(mooseVariable(), type, qrule->get_order());
        auto basis = _assembly.getBasis(tuple);
        auto rstr_elem = getCEEDRestrictor().getElementRestriction(tuple);
        auto & op = _operators[pair];

        CeedOperatorCreate(_ceed, qf, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op);

        for (auto var : getMooseVariableDependencies())
        {
          auto tuple = std::make_tuple(var, type, qrule->get_order());
          auto rstr_interp = getCEEDRestrictor().getQuadratureRestriction(tuple, CEED_EVAL_INTERP);
          auto rstr_grad = getCEEDRestrictor().getQuadratureRestriction(tuple, CEED_EVAL_GRAD);

          for (auto tag : tags)
          {
            auto qp_interp =
                _systems[var->sys().number()].getQuadratureVector(tuple, tag, CEED_EVAL_INTERP);
            auto qp_grad =
                _systems[var->sys().number()].getQuadratureVector(tuple, tag, CEED_EVAL_GRAD);

            std::string name = var->name() + "_tag_" + std::to_string(tag);

            CeedOperatorSetField(op, name.c_str(), rstr_interp, CEED_BASIS_NONE, qp_interp);
            CeedOperatorSetField(op, (name + "_grad").c_str(), rstr_grad, CEED_BASIS_NONE, qp_grad);
          }
        }

        CeedOperatorSetField(op, "residual", rstr_elem, basis, CEED_VECTOR_ACTIVE);
      }
  }

  virtual const MooseVariable & variable() const override { return _var; }

  // Dispatch residual calculation to GPU
  virtual void computeResidual() override
  {
    auto kernel = static_cast<Kernel &>(*this);

    GPUMirror<Kernel> mirror(kernel);

    auto & system = _systems[_var.sys().number()];

    CeedQFunctionContextSetData(
        _ctx, CEED_MEM_DEVICE, CEED_USE_POINTER, sizeof(Kernel *), mirror.get());

    for (auto type : _assembly.getUniqueElemTypes())
      for (auto qrule : _assembly.getUniqueQuadratures())
      {
        auto pair = std::make_pair(type, qrule->get_order());
        auto & op = _operators[pair];

        for (auto tag : system.getActiveResidualTags())
        {
          auto & res = system.getVector(tag);

          CeedOperatorApplyAdd(op, CEED_VECTOR_NONE, res, CEED_REQUEST_IMMEDIATE);
        }
      }
  }

  // Dispatch Jacobian calculation to GPU
  virtual void computeJacobian() override {}

public:
  CEED_FUNCTION void
  computeResidualInternal(const CeedInt Q, const CeedScalar * const * in, CeedScalar * const * out)
  {
    auto kernel = static_cast<Kernel *>(this);

    const CeedScalar *u = in[0], *q_data = in[1];
    CeedScalar * v = out[0];

    CeedPragmaSIMD for (CeedInt qp = 0; qp < Q; qp++) v[qp] +=
        q_data[qp] * kernel->computeQpResidual(qp);
  }

private:
  // CEED Q-function context containing this
  CEEDQFunctionContext _ctx;
  // CEED Q-function
  std::map<std::pair<ElemType, unsigned int>, CEEDQFunction> _functions;
  // CEED operator
  std::map<std::pair<ElemType, unsigned int>, CEEDOperator> _operators;

protected:
  // Reference to MooseVariable
  MooseVariable & _var;
};

#endif
