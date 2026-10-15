#include "SOFIE/ROperator.hxx"

#include "bindings.hpp"

using namespace SOFIE;

void sofie_bindings::bind_roperator(py::module_ &m)
{
   // The base of all operators. py::smart_holder is what lets RModel::AddOperator take ownership of one
   py::class_<ROperator, py::smart_holder>(m, "ROperator");
}
