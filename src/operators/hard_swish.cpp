#include "SOFIE/ROperator_HardSwish.hxx"

#include "bindings.hpp"

using namespace SOFIE;

void sofie_bindings::bind_hard_swish(py::module_ &m)
{
   bind_unary<ROperator_HardSwish<float>>(m, "ROperator_HardSwish_float");
}
