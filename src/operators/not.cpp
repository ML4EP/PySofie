#include "SOFIE/ROperator_Not.hxx"

#include "bindings.hpp"

using namespace SOFIE;

void sofie_bindings::bind_not(py::module_ &m)
{
   bind_unary<ROperator_Not>(m, "ROperator_Not");
}
