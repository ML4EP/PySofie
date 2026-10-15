#include "SOFIE/ROperator_Swish.hxx"

#include "bindings.hpp"

using namespace SOFIE;

void sofie_bindings::bind_swish(py::module_ &m)
{
   bind_unary<ROperator_Swish>(m, "ROperator_Swish");
}
